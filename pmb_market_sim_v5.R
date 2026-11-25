# =============================================================================
# pmb_market_sim_v5.R
# Budget-constrained programmatic advertising simulation
#
# George Ray
#
# Extends pmb_market_sim_gmr_v3_svm.R:
#   * n_auctions is the opportunity horizon.
#   * impression_budget is the maximum number of equal-cost impressions.
#   * Every opportunity reveals its audience segment before a bid decision.
#   * Fixed/diagonal campaigns have one eligible creative per segment.
#   * Rotation/crossed campaigns have multiple creatives per segment.
#   * Learners may preserve budget by declining low-predicted-value auctions.
#
# Primary systems:
#   random_exhaust     Bid on every auction until budget exhaustion.
#   random_paced       Spend randomly but pace across the full horizon.
#   ridge              Logistic Ridge + adaptive bid pacing.
#   two_way            Logistic Two-way + adaptive bid pacing.
#   thompson           Sequential Beta-Bernoulli Thompson sampling with the
#                      common proportional creative-allocation policy.
#   svm                Linear probabilistic SVM + adaptive bid pacing.
#   multilevel         Empirical-Bayes multilevel logistic partial pooling.
#   neural             Small feed-forward neural network + adaptive bid pacing.
#   oracle_online      Knows truth now, but not the future auction sequence.
#   oracle_clairvoyant Knows truth and the full future auction sequence.
#
# V5 preserves V4 simulation mathematics and seed formulas. Independent
# replications use Windows-compatible PSOCK workers; only the caller writes
# SQLite. workers = 1L selects serial execution.
#
# V5 sources V2 for shared truth generation, eligibility, model fitting, and
# creative allocation. Keep both files in the same directory.
# =============================================================================

# Remember this source file so clean PSOCK sessions load the same definitions.
.pmb_v5_source_path <- local({
  paths <- lapply(sys.frames(), function(frame) frame$ofile)
  paths <- Filter(function(path) is.character(path) && length(path) == 1L,
                  paths)
  normalizePath(tail(paths, 1L)[[1L]], mustWork = TRUE)
})
v2_candidates <- c(
  file.path(dirname(.pmb_v5_source_path), "pmb_market_sim_gmr_v2.R"),
  "C:/R_Code/pmb_market_sim_gmr_v2.R",
  file.path(getwd(), "pmb_market_sim_gmr_v2.R")
)
v2_path <- v2_candidates[file.exists(v2_candidates)][1]
if (is.na(v2_path)) {
  stop("Could not locate pmb_market_sim_gmr_v2.R.")
}
source(v2_path, local = TRUE)

nonconverged_refits <- 0L
final_refit_converged <- NA

# =============================================================================
# V4 ADDITIONAL LEARNERS: NEURAL, SVM, AND EMPIRICAL-BAYES MULTILEVEL
# =============================================================================

fit_neural_network <- function(
  history,
  prediction_cells,
  structure,
  segment_levels,
  creative_levels,
  hidden_units = 3L,
  decay = 0.001,
  maxit = 200L,
  min_conversions = 5L
) {
  diagnostics <- list(
    method = "neural",
    converged = FALSE,
    message = NULL,
    lambda = NA_real_
  )

  if (!requireNamespace("nnet", quietly = TRUE)) {
    diagnostics$message <- "Package 'nnet' is required for the neural learner."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }
  if (sum(history$converted) < min_conversions ||
      sum(1L - history$converted) < min_conversions) {
    diagnostics$message <- "Insufficient outcome variation."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }

  tryCatch({
    history <- prepare_model_data(
      history,
      segment_levels,
      creative_levels
    )
    prediction_cells <- prepare_model_data(
      prediction_cells,
      segment_levels,
      creative_levels
    )

    # Aggregate repeated input patterns before each refit. Frequency weights
    # preserve the Bernoulli cross-entropy contribution of all impressions.
    response_terms <- cbind(
      conversions = as.numeric(history$converted),
      impressions = rep(1, nrow(history))
    )
    if (structure == "diagonal") {
      cell_history <- aggregate(
        response_terms ~ segment, data = history, FUN = sum
      )
      model_formula <- ~ segment - 1
    } else if (structure == "crossed") {
      cell_history <- aggregate(
        response_terms ~ segment + creative, data = history, FUN = sum
      )
      model_formula <- ~ segment + creative - 1
    } else {
      stop("Unknown market structure for neural learner.")
    }
    cell_history$response <-
      cell_history$conversions / cell_history$impressions
    X <- model.matrix(model_formula, data = cell_history)
    X_pred <- model.matrix(model_formula, data = prediction_cells)
    if (!identical(colnames(X), colnames(X_pred))) {
      stop("Neural training and prediction matrices have different columns.")
    }

    fit <- nnet::nnet(
      x = X,
      y = cell_history$response,
      weights = cell_history$impressions,
      size = as.integer(hidden_units),
      decay = decay,
      maxit = as.integer(maxit),
      entropy = TRUE,
      linout = FALSE,
      trace = FALSE,
      MaxNWts = 1000L
    )

    predicted <- as.numeric(predict(fit, X_pred, type = "raw"))
    predicted <- pmin(pmax(predicted, 1e-6), 1 - 1e-6)
    diagnostics$converged <- fit$convergence == 0L
    if (!diagnostics$converged) {
      diagnostics$message <- "Neural optimizer reached maxit before convergence."
    }

    list(
      predictions = data.frame(
        key = prediction_cells$key,
        predicted = predicted,
        stringsAsFactors = FALSE
      ),
      diagnostics = diagnostics
    )
  }, error = function(e) {
    diagnostics$message <- conditionMessage(e)
    list(predictions = NULL, diagnostics = diagnostics)
  })
}

fit_linear_svm <- function(
  history,
  prediction_cells,
  structure,
  segment_levels,
  creative_levels,
  min_conversions = 5L,
  cost = 1
) {
  diagnostics <- list(
    method = "svm",
    converged = FALSE,
    message = NULL,
    lambda = NA_real_
  )

  if (sum(history$converted) < min_conversions ||
      sum(1L - history$converted) < min_conversions) {
    diagnostics$message <- "Insufficient outcome variation."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }
  if (length(cost) != 1L || !is.finite(cost) || cost <= 0) {
    diagnostics$message <- "svm_cost must be a finite positive scalar."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }

  tryCatch({
    prediction_cells <- prepare_model_data(
      prediction_cells,
      segment_levels,
      creative_levels
    )
    model_formula <- ridge_formula_for_structure(structure)
    X_pred <- model.matrix(model_formula, data = prediction_cells)
    history_key <- make_key(history$segment, history$creative)
    cell_index <- match(history_key, prediction_cells$key)
    if (anyNA(cell_index)) {
      stop("SVM history contains a cell outside the prediction action space.")
    }

    positives <- tabulate(
      cell_index[history$converted == 1L],
      nbins = nrow(prediction_cells)
    )
    totals <- tabulate(cell_index, nbins = nrow(prediction_cells))
    negatives <- totals - positives

    positives <- as.numeric(positives)
    negatives <- as.numeric(negatives)

    n_positive <- sum(positives)
    n_negative <- sum(negatives)
    n_history <- n_positive + n_negative

    # The predictor matrix contains only 8 diagonal or 64 crossed unique
    # rows. Repeated binary observations are therefore represented exactly by
    # their cell-level positive and negative counts in the squared-hinge loss.
    positive_weight <- positives * n_history / (2 * n_positive)
    negative_weight <- negatives * n_history / (2 * n_negative)

    if (any(!is.finite(positive_weight)) ||
        any(!is.finite(negative_weight))) {
        stop("SVM observation weights are not finite.")
    }

    weight_total <- sum(positive_weight) + sum(negative_weight)

    objective_gradient <- function(theta) {
      intercept <- theta[1L]
      coefficients <- theta[-1L]
      score <- as.numeric(intercept + X_pred %*% coefficients)
      positive_hinge <- pmax(0, 1 - score)
      negative_hinge <- pmax(0, 1 + score)
      weighted_loss <- sum(
        positive_weight * positive_hinge^2 +
          negative_weight * negative_hinge^2
      ) / weight_total
      objective <- 0.5 * sum(coefficients^2) + cost * weighted_loss

      score_gradient <- cost * (
        -2 * positive_weight * positive_hinge +
          2 * negative_weight * negative_hinge
      ) / weight_total
      gradient <- c(
        sum(score_gradient),
        coefficients + as.numeric(crossprod(X_pred, score_gradient))
      )
      list(objective = objective, gradient = gradient)
    }

    initial <- numeric(ncol(X_pred) + 1L)
    fit <- optim(
      par = initial,
      fn = function(theta) objective_gradient(theta)$objective,
      gr = function(theta) objective_gradient(theta)$gradient,
      method = "BFGS",
      control = list(maxit = 500L, reltol = 1e-9)
    )
    svm_score <- as.numeric(fit$par[1L] + X_pred %*% fit$par[-1L])

    # Platt-style calibration converts margin scores to probabilities. The
    # original, unbalanced counts are used so calibration retains the actual
    # campaign conversion base rate after class-balanced SVM estimation.
    calibration <- suppressWarnings(glm(
      cbind(positives, negatives) ~ svm_score,
      family = binomial()
    ))
    calibration_coefficients <- coef(calibration)
    if (any(!is.finite(calibration_coefficients))) {
      stop("SVM probability calibration failed.")
    }
    predicted <- as.numeric(plogis(
      calibration_coefficients[1L] +
        calibration_coefficients[2L] * svm_score
    ))
    if (any(!is.finite(predicted)) ||
        any(predicted < 0) || any(predicted > 1)) {
      stop("SVM produced invalid conversion probabilities.")
    }

    diagnostics$converged <- fit$convergence == 0L
    diagnostics$message <- if (fit$convergence == 0L) {
      NULL
    } else {
      paste("SVM optimizer convergence code", fit$convergence)
    }
    list(
      predictions = data.frame(
        key = prediction_cells$key,
        predicted = predicted,
        stringsAsFactors = FALSE
      ),
      diagnostics = diagnostics
    )
  }, error = function(e) {
    diagnostics$message <- conditionMessage(e)
    list(predictions = NULL, diagnostics = diagnostics)
  })
}

fit_multilevel_logistic <- function(
  history,
  prediction_cells,
  structure,
  segment_levels,
  creative_levels,
  min_conversions = 5L
) {
  diagnostics <- list(
    method = "multilevel",
    converged = FALSE,
    message = NULL,
    lambda = NA_real_
  )

  if (!requireNamespace("lme4", quietly = TRUE)) {
    diagnostics$message <-
      "Package 'lme4' is required for the multilevel learner."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }
  if (sum(history$converted) < min_conversions ||
      sum(1L - history$converted) < min_conversions) {
    diagnostics$message <- "Insufficient outcome variation."
    return(list(predictions = NULL, diagnostics = diagnostics))
  }

  warning_messages <- character()
  tryCatch(
    withCallingHandlers({
      prediction_cells <- prepare_model_data(
        prediction_cells,
        segment_levels,
        creative_levels
      )
      prediction_cells$cell <- factor(
        prediction_cells$key,
        levels = prediction_cells$key
      )

      history_key <- make_key(history$segment, history$creative)
      cell_index <- match(history_key, prediction_cells$key)
      if (anyNA(cell_index)) {
        stop(
          "Multilevel history contains a cell outside the prediction space."
        )
      }
      conversions <- tabulate(
        cell_index[history$converted == 1L],
        nbins = nrow(prediction_cells)
      )
      totals <- tabulate(cell_index, nbins = nrow(prediction_cells))
      cell_history <- prediction_cells[, c(
        "key", "segment", "creative", "cell"
      ), drop = FALSE]
      cell_history$conversions <- conversions
      cell_history$nonconversions <- totals - conversions
      cell_history <- cell_history[totals > 0L, , drop = FALSE]

      model_formula <- if (structure == "diagonal") {
        cbind(conversions, nonconversions) ~ 1 + (1 | cell)
      } else if (structure == "crossed") {
        cbind(conversions, nonconversions) ~
          1 + (1 | segment) + (1 | creative) + (1 | cell)
      } else {
        stop("Unknown market structure for the multilevel learner.")
      }

      fit <- lme4::glmer(
        formula = model_formula,
        data = cell_history,
        family = binomial(),
        nAGQ = 1L,
        control = lme4::glmerControl(
          optimizer = "bobyqa",
          calc.derivs = FALSE,
          check.nobs.vs.nRE = "ignore",
          check.nobs.vs.nlev = "ignore"
        )
      )
      predicted <- as.numeric(predict(
        fit,
        newdata = prediction_cells,
        type = "response",
        re.form = NULL,
        allow.new.levels = TRUE
      ))
      if (any(!is.finite(predicted)) ||
          any(predicted < 0) || any(predicted > 1)) {
        stop("Multilevel model produced invalid conversion probabilities.")
      }

      convergence_messages <- fit@optinfo$conv$lme4$messages
      singular <- lme4::isSingular(fit, tol = 1e-4)
      diagnostics$converged <- is.null(convergence_messages)
      diagnostic_messages <- c(
        warning_messages,
        convergence_messages,
        if (singular) "Singular multilevel fit." else NULL
      )
      diagnostics$message <- if (length(diagnostic_messages)) {
        paste(unique(diagnostic_messages), collapse = " | ")
      } else {
        NULL
      }

      list(
        predictions = data.frame(
          key = prediction_cells$key,
          predicted = predicted,
          stringsAsFactors = FALSE
        ),
        diagnostics = diagnostics
      )
    }, warning = function(w) {
      warning_messages <<- c(warning_messages, conditionMessage(w))
      invokeRestart("muffleWarning")
    }),
    error = function(e) {
      diagnostics$message <- conditionMessage(e)
      list(predictions = NULL, diagnostics = diagnostics)
    }
  )
}

fit_learner_v4 <- function(
  method,
  history,
  prediction_cells,
  structure,
  segment_levels,
  creative_levels,
  svm_cost = 1,
  neural_hidden_units = 3L,
  neural_decay = 0.001,
  neural_maxit = 1000L
) {
  if (method == "neural") {
    fit_neural_network(
      history = history,
      prediction_cells = prediction_cells,
      structure = structure,
      segment_levels = segment_levels,
      creative_levels = creative_levels,
      hidden_units = neural_hidden_units,
      decay = neural_decay,
      maxit = neural_maxit
    )
  } else if (method == "svm") {
    fit_linear_svm(
      history = history,
      prediction_cells = prediction_cells,
      structure = structure,
      segment_levels = segment_levels,
      creative_levels = creative_levels,
      cost = svm_cost
    )
  } else if (method == "multilevel") {
    fit_multilevel_logistic(
      history = history,
      prediction_cells = prediction_cells,
      structure = structure,
      segment_levels = segment_levels,
      creative_levels = creative_levels
    )
  } else {
    fit_learner(
      method = method,
      history = history,
      prediction_cells = prediction_cells,
      structure = structure,
      segment_levels = segment_levels,
      creative_levels = creative_levels
    )
  }
}

# =============================================================================
# 1. AUCTION ENVIRONMENT
# =============================================================================

make_auction_environment <- function(
  truth,
  n_auctions,
  segment_probabilities = NULL
) {
  if (n_auctions < 1L) {
    stop("n_auctions must be positive.")
  }

  environment <- make_replication_environment(
    truth = truth,
    n_impressions = n_auctions,
    segment_probabilities = segment_probabilities
  )
  names(environment)[names(environment) == "segment_arrivals"] <-
    "auction_segments"
  environment$n_auctions <- n_auctions
  environment
}

validate_budget_design <- function(
  n_auctions,
  impression_budget,
  impression_cost
) {
  if (length(n_auctions) != 1L || !is.finite(n_auctions) ||
      n_auctions < 1 || n_auctions != as.integer(n_auctions)) {
    stop("n_auctions must be a positive integer.")
  }
  if (length(impression_budget) != 1L ||
      !is.finite(impression_budget) ||
      impression_budget < 1 ||
      impression_budget != as.integer(impression_budget)) {
    stop("impression_budget must be a positive integer.")
  }
  if (impression_budget > n_auctions) {
    stop("impression_budget cannot exceed n_auctions.")
  }
  if (length(impression_cost) != 1L ||
      !is.finite(impression_cost) ||
      impression_cost <= 0) {
    stop("impression_cost must be positive.")
  }
  invisible(TRUE)
}

# =============================================================================
# 2. CREATIVE AND BID DECISIONS
# =============================================================================

get_segment_cells <- function(params, segment) {
  cells <- params$cells[
    params$cells$segment == segment,
    ,
    drop = FALSE
  ]
  if (nrow(cells) == 0L) {
    stop(sprintf("No eligible creative for segment '%s'.", segment))
  }
  cells
}

lookup_prediction <- function(predictions, key) {
  if (is.null(predictions)) {
    return(NA_real_)
  }
  value <- predictions$predicted[match(key, predictions$key)]
  if (length(value) != 1L || !is.finite(value)) {
    NA_real_
  } else {
    value
  }
}

draw_thompson_predictions <- function(cells, alpha, beta) {
  keys <- as.character(cells$key)
  if (!all(keys %in% names(alpha)) || !all(keys %in% names(beta))) {
    stop("Thompson posterior does not contain every eligible cell key.")
  }
  data.frame(
    key = keys,
    predicted = stats::rbeta(
      length(keys),
      shape1 = unname(alpha[keys]),
      shape2 = unname(beta[keys])
    ),
    stringsAsFactors = FALSE
  )
}

select_creative_v4 <- function(
  eligible_cells,
  system,
  predictions = NULL,
  epsilon = 0.10,
  allocation_policy = c("proportional", "epsilon_greedy")
) {
  allocation_policy <- match.arg(allocation_policy)

  if (nrow(eligible_cells) == 1L) {
    selected <- eligible_cells
    selected$was_creative_exploration <- FALSE
    selected$creative_selection_probability <- 1
    return(selected)
  }

  if (system %in% c("oracle_online", "oracle_clairvoyant")) {
    selected <- allocate_oracle(eligible_cells)
    names(selected)[names(selected) == "was_exploration"] <-
      "was_creative_exploration"
    names(selected)[names(selected) == "selection_probability"] <-
      "creative_selection_probability"
    return(selected)
  }

  if (system %in%
      c("ridge", "two_way", "svm", "thompson", "multilevel", "neural") &&
      !is.null(predictions)) {
    selected <- allocate_from_predictions(
      eligible_cells = eligible_cells,
      predictions = predictions,
      epsilon = epsilon,
      policy = allocation_policy
    )
    names(selected)[names(selected) == "was_exploration"] <-
      "was_creative_exploration"
    names(selected)[names(selected) == "selection_probability"] <-
      "creative_selection_probability"
    return(selected)
  }

  selected <- allocate_random(eligible_cells)
  selected$was_creative_exploration <-
    system %in% c(
      "ridge", "two_way", "svm", "thompson", "multilevel", "neural"
    )
  selected$creative_selection_probability <- 1 / nrow(eligible_cells)
  selected
}

true_segment_scores <- function(params) {
  scores <- aggregate(
    true_p ~ segment,
    data = params$cells,
    FUN = max
  )
  setNames(scores$true_p, scores$segment)
}

predicted_segment_scores <- function(params, predictions) {
  if (is.null(predictions)) {
    return(NULL)
  }
  joined <- merge(
    params$cells[, c("segment", "key"), drop = FALSE],
    predictions[, c("key", "predicted"), drop = FALSE],
    by = "key",
    all.x = TRUE,
    sort = FALSE
  )
  if (anyNA(joined$predicted) || any(!is.finite(joined$predicted))) {
    return(NULL)
  }
  scores <- aggregate(predicted ~ segment, data = joined, FUN = max)
  setNames(scores$predicted, scores$segment)
}

# Returns a bid probability that accepts approximately the best q proportion
# of opportunities, accounting for the segment supply distribution. Fractional
# acceptance at the threshold handles discrete segment categories.
selective_bid_probability <- function(
  current_score,
  segment_scores,
  segment_probabilities,
  target_acceptance_rate
) {
  q <- min(1, max(0, target_acceptance_rate))
  if (q <= 0) return(0)
  if (q >= 1) return(1)
  if (!is.finite(current_score) ||
      is.null(segment_scores) ||
      any(!is.finite(segment_scores))) {
    return(q)
  }

  probabilities <- segment_probabilities[names(segment_scores)]
  if (anyNA(probabilities)) {
    stop("Segment score names do not match segment probabilities.")
  }
  probabilities <- probabilities / sum(probabilities)

  greater_mass <- sum(probabilities[segment_scores > current_score])
  equal_mass <- sum(probabilities[segment_scores == current_score])

  if (q <= greater_mass) {
    0
  } else if (q >= greater_mass + equal_mass) {
    1
  } else {
    (q - greater_mass) / equal_mass
  }
}

make_clairvoyant_bid_schedule <- function(
  params,
  environment,
  impression_budget
) {
  scores <- true_segment_scores(params)
  auction_scores <- unname(scores[environment$auction_segments])

  # Random tie-breaking prevents systematic preference for early or late
  # auctions when many opportunities have identical segment-level truth.
  tie_breaker <- runif(length(auction_scores))
  selected <- order(-auction_scores, tie_breaker)[
    seq_len(impression_budget)
  ]
  schedule <- rep(FALSE, length(auction_scores))
  schedule[selected] <- TRUE
  schedule
}

# =============================================================================
# 3. BUDGET-CONSTRAINED CAMPAIGN
# =============================================================================

simulate_budget_campaign <- function(
  params,
  environment,
  system = c(
    "random_exhaust",
    "random_paced",
    "ridge",
    "two_way",
    "thompson",
    "svm",
    "multilevel",
    "neural",
    "oracle_online",
    "oracle_clairvoyant"
  ),
  impression_budget = 155000L,
  impression_cost = 1,
  burnin_impressions = 10000L,
  refit_interval = 5000L,
  creative_epsilon = 0.10,
  bid_epsilon = 0.05,
  allocation_policy = c("proportional", "epsilon_greedy"),
  thompson_prior_alpha = 1,
  thompson_prior_beta = 1,
  svm_cost = 1,
  neural_hidden_units = 3L,
  neural_decay = 0.001,
  neural_maxit = 1000L,
  keep_event_log = FALSE,
  verbose = FALSE
) {
  system <- match.arg(system)
  allocation_policy <- match.arg(allocation_policy)
  n_auctions <- environment$n_auctions

  validate_budget_design(
    n_auctions,
    impression_budget,
    impression_cost
  )
  if (burnin_impressions < 0L || refit_interval < 1L) {
    stop("burnin_impressions must be nonnegative and refit_interval positive.")
  }
  if (creative_epsilon < 0 || creative_epsilon > 1 ||
      bid_epsilon < 0 || bid_epsilon > 1) {
    stop("Exploration rates must lie between zero and one.")
  }
  if (length(thompson_prior_alpha) != 1L ||
      length(thompson_prior_beta) != 1L ||
      !is.finite(thompson_prior_alpha) ||
      !is.finite(thompson_prior_beta) ||
      thompson_prior_alpha <= 0 || thompson_prior_beta <= 0) {
    stop("Thompson prior parameters must be finite positive scalars.")
  }
  if (length(svm_cost) != 1L || !is.finite(svm_cost) || svm_cost <= 0) {
    stop("svm_cost must be a finite positive scalar.")
  }
  if (length(neural_hidden_units) != 1L ||
      !is.finite(neural_hidden_units) || neural_hidden_units < 1L ||
      neural_hidden_units != as.integer(neural_hidden_units)) {
    stop("neural_hidden_units must be a positive integer.")
  }
  if (length(neural_decay) != 1L ||
      !is.finite(neural_decay) || neural_decay < 0) {
    stop("neural_decay must be finite and nonnegative.")
  }
  if (length(neural_maxit) != 1L || !is.finite(neural_maxit) ||
      neural_maxit < 1L || neural_maxit != as.integer(neural_maxit)) {
    stop("neural_maxit must be a positive integer.")
  }

  remaining_impressions <- as.integer(impression_budget)
  remaining_budget <- impression_budget * impression_cost
  purchased <- 0L
  total_conversions <- 0L
  expected_conversions <- 0
  cumulative_creative_regret <- 0
  predictions <- NULL
  model_segment_scores <- NULL
  thompson_alpha <- setNames(
    rep(thompson_prior_alpha, nrow(params$cells)),
    params$cells$key
  )
  thompson_beta <- setNames(
    rep(thompson_prior_beta, nrow(params$cells)),
    params$cells$key
  )
  thompson_segment_indices <- split(
    seq_len(nrow(params$cells)),
    factor(params$cells$segment, levels = params$segments)
  )
  thompson_segment_scores <- NULL
  fit_failures <- 0L
  fit_messages <- character()
  lambda_track <- numeric()

  # Per-campaign neural convergence diagnostics
  nonconverged_refits <- 0L
  final_refit_converged <- NA

  next_refit <- max(1L, as.integer(burnin_impressions))
  exhausted_at <- NA_integer_

  # Only purchased impressions enter supervised training history.
  history_segment <- character(impression_budget)
  history_creative <- character(impression_budget)
  history_converted <- integer(impression_budget)

  if (keep_event_log) {
    event_segment <- environment$auction_segments
    event_bid <- logical(n_auctions)
    event_creative <- rep(NA_character_, n_auctions)
    event_key <- rep(NA_character_, n_auctions)
    event_predicted <- rep(NA_real_, n_auctions)
    event_bid_probability <- numeric(n_auctions)
    event_creative_probability <- rep(NA_real_, n_auctions)
    event_creative_exploration <- logical(n_auctions)
    event_true_p <- rep(NA_real_, n_auctions)
    event_converted <- rep(NA_integer_, n_auctions)
    event_remaining_impressions <- integer(n_auctions)
  }

  clairvoyant_schedule <- if (system == "oracle_clairvoyant") {
    make_clairvoyant_bid_schedule(
      params,
      environment,
      impression_budget
    )
  } else {
    NULL
  }
  truth_scores <- if (system == "oracle_online") {
    true_segment_scores(params)
  } else {
    NULL
  }

  for (auction in seq_len(n_auctions)) {
    remaining_auctions <- n_auctions - auction + 1L

    if (remaining_impressions <= 0L) {
      if (is.na(exhausted_at)) exhausted_at <- auction - 1L
      if (keep_event_log) {
        event_remaining_impressions[auction:n_auctions] <- 0L
      }
      break
    }

    segment <- environment$auction_segments[auction]
    eligible_cells <- get_segment_cells(params, segment)

    target_rate <- min(
      1,
      remaining_impressions / remaining_auctions
    )

    # Thompson draws score the segment for bidding. If the auction is accepted,
    # the common allocation policy selects among the eligible creatives using
    # these draws as its proportional weights. The served creative can therefore
    # differ from the creative whose maximum draw justified the bid.
    if (system == "thompson") {
      predictions <- draw_thompson_predictions(
        params$cells,
        thompson_alpha,
        thompson_beta
      )
      thompson_segment_scores <- vapply(
        thompson_segment_indices,
        function(indices) max(predictions$predicted[indices]),
        numeric(1)
      )
    }

    if (system == "random_exhaust") {
      bid_probability <- 1
      bid <- TRUE
    } else if (system == "random_paced") {
      bid_probability <- target_rate
      bid <- runif(1) < bid_probability
    } else if (system == "oracle_clairvoyant") {
      bid_probability <- as.numeric(clairvoyant_schedule[auction])
      bid <- clairvoyant_schedule[auction]
    } else {
      segment_scores <- if (system == "oracle_online") {
        truth_scores
      } else if (system == "thompson") {
        thompson_segment_scores
      } else {
        model_segment_scores
      }

      current_score <- if (is.null(segment_scores)) {
        NA_real_
      } else {
        unname(segment_scores[segment])
      }

      selective_probability <- selective_bid_probability(
        current_score = current_score,
        segment_scores = segment_scores,
        segment_probabilities = environment$segment_probabilities,
        target_acceptance_rate = target_rate
      )

      # Blend selective bidding with random pacing. This preserves exploration
      # and keeps the expected overall acceptance rate close to target_rate.
      bid_probability <- (1 - bid_epsilon) * selective_probability +
        bid_epsilon * target_rate

      # If every remaining auction is needed to spend the budget, force a bid.
      if (remaining_impressions >= remaining_auctions) {
        bid_probability <- 1
      }
      bid <- runif(1) < bid_probability
    }

    if (keep_event_log) {
      event_bid[auction] <- bid
      event_bid_probability[auction] <- bid_probability
      event_remaining_impressions[auction] <- remaining_impressions
    }

    if (!bid) {
      next
    }

    selected <- select_creative_v4(
      eligible_cells = eligible_cells,
      system = system,
      predictions = predictions,
      epsilon = creative_epsilon,
      allocation_policy = allocation_policy
    )

    predicted_p <- lookup_prediction(predictions, selected$key)
    outcome_u <- environment$outcome_uniforms[auction, selected$key]
    converted <- as.integer(outcome_u < selected$true_p)

    purchased <- purchased + 1L
    remaining_impressions <- remaining_impressions - 1L
    remaining_budget <- remaining_budget - impression_cost
    total_conversions <- total_conversions + converted
    expected_conversions <- expected_conversions + selected$true_p
    cumulative_creative_regret <- cumulative_creative_regret +
      selected$oracle_regret

    history_segment[purchased] <- as.character(selected$segment)
    history_creative[purchased] <- as.character(selected$creative)
    history_converted[purchased] <- converted

    if (system == "thompson") {
      selected_key <- as.character(selected$key)
      thompson_alpha[selected_key] <-
        thompson_alpha[selected_key] + converted
      thompson_beta[selected_key] <-
        thompson_beta[selected_key] + 1L - converted
    }

    if (keep_event_log) {
      event_creative[auction] <- as.character(selected$creative)
      event_key[auction] <- as.character(selected$key)
      event_predicted[auction] <- predicted_p
      event_creative_probability[auction] <-
        selected$creative_selection_probability
      event_creative_exploration[auction] <-
        selected$was_creative_exploration
      event_true_p[auction] <- selected$true_p
      event_converted[auction] <- converted
      event_remaining_impressions[auction] <- remaining_impressions
    }

    should_refit <- system %in%
      c("ridge", "two_way", "svm", "multilevel", "neural") &&
      purchased >= next_refit

    if (should_refit) {
      history <- data.frame(
        segment = history_segment[seq_len(purchased)],
        creative = history_creative[seq_len(purchased)],
        converted = history_converted[seq_len(purchased)],
        stringsAsFactors = FALSE
      )

      fit_result <- fit_learner_v4(
        method = system,
        history = history,
        prediction_cells = params$cells,
        structure = params$structure,
        segment_levels = params$segments,
        creative_levels = params$creatives,
        svm_cost = svm_cost,
        neural_hidden_units = neural_hidden_units,
        neural_decay = neural_decay,
        neural_maxit = neural_maxit
      )

# Record neural convergence for this individual refit.
if (system == "neural") {
  current_converged <-
    isTRUE(fit_result$diagnostics$converged)

  if (!current_converged) {
    nonconverged_refits <-
      nonconverged_refits + 1L
  }

  # This value is overwritten after every refit, so when the campaign
  # finishes it describes the final neural refit.
  final_refit_converged <- current_converged
}

      if (!is.null(fit_result$predictions)) {
        predictions <- fit_result$predictions
        model_segment_scores <- predicted_segment_scores(
          params,
          predictions
        )
      } else {
        fit_failures <- fit_failures + 1L
      }
      if (is.finite(fit_result$diagnostics$lambda)) {
        lambda_track <- c(
          lambda_track,
          fit_result$diagnostics$lambda
        )
      }
      if (!is.null(fit_result$diagnostics$message)) {
        fit_messages <- c(
          fit_messages,
          fit_result$diagnostics$message
        )
      }

      if (verbose) {
        cat(sprintf(
          "  auction=%7d | bought=%6d | %-8s/%-8s | conversions=%d\n",
          auction,
          purchased,
          params$structure,
          system,
          total_conversions
        ))
      }

      repeat {
        next_refit <- next_refit + refit_interval
        if (next_refit > purchased) break
      }
    }

    if (remaining_impressions == 0L && is.na(exhausted_at)) {
      exhausted_at <- auction
    }
  }

  result <- list(
    market_structure = params$structure,
    system = system,
    allocation_policy = if (system %in% c(
      "ridge", "two_way", "svm", "thompson", "multilevel", "neural"
    )) {
      allocation_policy
    } else {
      system
    },
    n_auctions = n_auctions,
    impression_budget = impression_budget,
    impressions_purchased = purchased,
    impressions_rejected = n_auctions - purchased,
    impression_cost = impression_cost,
    starting_budget = impression_budget * impression_cost,
    total_spend = purchased * impression_cost,
    remaining_budget = remaining_budget,
    budget_utilization = purchased / impression_budget,
    auction_acceptance_rate = purchased / n_auctions,
    budget_exhausted_at = exhausted_at,
    total_conversions = total_conversions,
    conversion_rate = if (purchased > 0L) {
      total_conversions / purchased
    } else {
      NA_real_
    },
    conversions_per_auction = total_conversions / n_auctions,
    expected_conversions = expected_conversions,
    expected_conversion_rate = if (purchased > 0L) {
      expected_conversions / purchased
    } else {
      NA_real_
    },
    cost_per_conversion = if (total_conversions > 0L) {
      purchased * impression_cost / total_conversions
    } else {
      Inf
    },
    cumulative_creative_regret = cumulative_creative_regret,
    mean_lambda = if (length(lambda_track)) {
      mean(lambda_track)
    } else {
      NA_real_
    },
    final_lambda = if (length(lambda_track)) {
      tail(lambda_track, 1L)
    } else {
      NA_real_
    },
    fit_failures = fit_failures,
    fit_messages = paste(unique(fit_messages), collapse = " | "),

    nonconverged_refits = nonconverged_refits,
    final_refit_converged = final_refit_converged

  )

  if (keep_event_log) {
    result$events <- data.frame(
      auction = seq_len(n_auctions),
      segment = event_segment,
      bid = event_bid,
      bid_probability = event_bid_probability,
      creative = event_creative,
      key = event_key,
      predicted = event_predicted,
      creative_selection_probability = event_creative_probability,
      was_creative_exploration = event_creative_exploration,
      true_p = event_true_p,
      converted = event_converted,
      remaining_impressions = event_remaining_impressions,
      stringsAsFactors = FALSE
    )
  }

  result
}

# =============================================================================
# 4. SQLITE OUTPUT
# =============================================================================

init_v4_results_db <- function(db_path = "pmb_market_v4.db") {
  if (!requireNamespace("RSQLite", quietly = TRUE)) {
    stop("Package 'RSQLite' is required for database output.")
  }
  con <- RSQLite::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(RSQLite::dbDisconnect(con), add = TRUE)

  RSQLite::dbExecute(con, "
    CREATE TABLE IF NOT EXISTS auction_run_summary (
      replication               INTEGER NOT NULL,
      market_structure          TEXT NOT NULL,
      system                    TEXT NOT NULL,
      allocation_policy         TEXT NOT NULL,
      n_auctions                INTEGER NOT NULL,
      impression_budget         INTEGER NOT NULL,
      impressions_purchased     INTEGER NOT NULL,
      impression_cost           REAL NOT NULL,
      starting_budget           REAL NOT NULL,
      total_spend               REAL NOT NULL,
      remaining_budget          REAL NOT NULL,
      budget_utilization        REAL NOT NULL,
      auction_acceptance_rate   REAL NOT NULL,
      budget_exhausted_at       INTEGER,
      total_conversions         INTEGER NOT NULL,
      conversion_rate           REAL,
      conversions_per_auction   REAL NOT NULL,
      expected_conversions      REAL NOT NULL,
      expected_conversion_rate REAL,
      cost_per_conversion       REAL,
      lift_over_random_exhaust  REAL,
      incremental_conversions   REAL,
      cumulative_creative_regret REAL NOT NULL,
      mean_lambda               REAL,
      final_lambda              REAL,
      fit_failures              INTEGER NOT NULL,
      fit_messages              TEXT,
      timestamp                 TEXT DEFAULT (datetime('now')),
      PRIMARY KEY (
        replication,
        market_structure,
        system,
        allocation_policy
      )
    )")

  # Reuse the corrected V2 cell-truth schema in this separate V4 database.
  RSQLite::dbExecute(con, "
    CREATE TABLE IF NOT EXISTS cell_ground_truth (
      replication       INTEGER NOT NULL,
      market_structure  TEXT NOT NULL,
      cell_key          TEXT NOT NULL,
      segment           TEXT NOT NULL,
      creative          TEXT NOT NULL,
      eligible          INTEGER NOT NULL,
      true_p            REAL NOT NULL,
      true_logit        REAL NOT NULL,
      segment_effect    REAL NOT NULL,
      creative_effect   REAL NOT NULL,
      interaction_effect REAL NOT NULL,
      is_segment_best   INTEGER NOT NULL,
      is_only_available INTEGER NOT NULL,
      rank_within_segment INTEGER NOT NULL,
      oracle_regret     REAL NOT NULL,
      PRIMARY KEY (replication, market_structure, cell_key)
    )")

  invisible(db_path)
}

write_v4_run_summary <- function(db_path, row) {
  con <- RSQLite::dbConnect(RSQLite::SQLite(), db_path)
  on.exit(RSQLite::dbDisconnect(con), add = TRUE)

  fields <- c(
    "replication", "market_structure", "system",
    "allocation_policy", "n_auctions", "impression_budget",
    "impressions_purchased", "impression_cost", "starting_budget",
    "total_spend", "remaining_budget", "budget_utilization",
    "auction_acceptance_rate", "budget_exhausted_at",
    "total_conversions", "conversion_rate", "conversions_per_auction",
    "expected_conversions", "expected_conversion_rate",
    "cost_per_conversion", "lift_over_random_exhaust",
    "incremental_conversions", "cumulative_creative_regret",
    "mean_lambda", "final_lambda", "fit_failures", "fit_messages"
  )

  RSQLite::dbWriteTable(
    con,
    "auction_run_summary",
    row[, fields, drop = FALSE],
    append = TRUE,
    row.names = FALSE
  )
}

# =============================================================================
# 5. V5 PARALLEL COMPARISON RUNNER
# =============================================================================

# A worker receives only simulation configuration, never a database path or
# connection. Keep both structures and all systems in one replication so they
# share exactly V4's truth, arrivals, and potential outcomes.
.pmb_v5_replication <- function(replication, config) {
  list2env(config, envir = environment())
  result_rows <- list()
  event_logs <- list()
  result_index <- 0L
  set.seed(as.integer(seed_base + replication))
  truth <- make_market_truth(
    base_rate = base_rate,
    segment_sd = segment_sd,
    creative_sd = creative_sd,
    interaction_sd = interaction_sd
  )
  diagonal <- make_diagonal_params(truth)
  crossed <- make_crossed_params(truth)
  validate_market_pair(diagonal, crossed)
  environment <- make_auction_environment(
    truth,
    n_auctions,
    segment_probabilities
  )

  for (params in list(diagonal, crossed)) {
    structure_rows <- list()

    for (system in systems) {
      system_code <- match(system, valid_systems)
      structure_code <- match(
        params$structure,
        c("diagonal", "crossed")
      )
      set.seed(as.integer(
        seed_base +
          replication * 10000L +
          structure_code * 1000L +
          system_code * 100L
      ))

      sim <- simulate_budget_campaign(
        params = params,
        environment = environment,
        system = system,
        impression_budget = impression_budget,
        impression_cost = impression_cost,
        burnin_impressions = burnin_impressions,
        refit_interval = refit_interval,
        creative_epsilon = creative_epsilon,
        bid_epsilon = bid_epsilon,
        allocation_policy = allocation_policy,
        thompson_prior_alpha = thompson_prior_alpha,
        thompson_prior_beta = thompson_prior_beta,
        svm_cost = svm_cost,
        neural_hidden_units = neural_hidden_units,
        neural_decay = neural_decay,
        neural_maxit = neural_maxit,
        keep_event_log = keep_event_log,
        verbose = verbose
      )

      row <- as.data.frame(
        sim[setdiff(names(sim), "events")],
        stringsAsFactors = FALSE
      )
      row$replication <- replication
      structure_rows[[system]] <- row

      if (keep_event_log) {
        log_name <- paste(
          replication,
          params$structure,
          system,
          sep = "/"
        )
        events <- sim$events
        events$replication <- replication
        events$market_structure <- params$structure
        events$system <- system
        event_logs[[log_name]] <- events
      }

      cat(sprintf(
        paste0(
          "  %-8s | %-19s: bought=%6d, conv=%6d, ",
          "CVR=%6.3f%%, budget=%6.2f%%\n"
        ),
        params$structure,
        system,
        sim$impressions_purchased,
        sim$total_conversions,
        100 * sim$conversion_rate,
        100 * sim$budget_utilization
      ))
    }

    structure_df <- do.call(rbind, structure_rows)
    baseline <- structure_df$total_conversions[
      structure_df$system == "random_exhaust"
    ]
    structure_df$incremental_conversions <-
      structure_df$total_conversions - baseline
    structure_df$lift_over_random_exhaust <- 100 *
      structure_df$incremental_conversions / baseline

    for (i in seq_len(nrow(structure_df))) {
      result_index <- result_index + 1L
      result_rows[[result_index]] <- structure_df[i, , drop = FALSE]
    }
  }
  list(rows = result_rows, events = event_logs,
       truth = list(diagonal, crossed))
}

# Top-level bootstrap avoids serializing the runner's frame (and db_path).
.pmb_v5_initialize_worker <- function(script, libs, rng) {
  .libPaths(libs)
  source(script, local = .GlobalEnv)
  do.call(RNGkind, as.list(rng))
  NULL
}

# Leave one CPU available by default; callers can explicitly choose workers.
.pmb_v5_default_workers <- function() {
  cores <- parallel::detectCores(logical = FALSE)
  if (is.na(cores)) 1L else max(1L, cores - 1L)
}

run_budget_market_comparison <- function(
  n_reps = 10L,
  n_auctions = 500000L,
  impression_budget = 155000L,
  impression_cost = 1,
  burnin_impressions = 10000L,
  refit_interval = 5000L,
  creative_epsilon = 0.10,
  bid_epsilon = 0.05,
  allocation_policy = c("proportional", "epsilon_greedy"),
  thompson_prior_alpha = 1,
  thompson_prior_beta = 1,
  svm_cost = 1,
  neural_hidden_units = 3L,
  neural_decay = 0.001,
  neural_maxit = 1000L,
  segment_probabilities = NULL,
  seed_base = 20260728L,
  base_rate = 0.03,
  segment_sd = 0.25,
  creative_sd = 0.20,
  interaction_sd = 0.30,
  systems = c(
    "random_exhaust",
    "random_paced",
    "ridge",
    "two_way",
    "thompson",
    "svm",
    "multilevel",
    "neural",
    "oracle_online",
    "oracle_clairvoyant"
  ),
  db_path = NULL,
  keep_event_log = FALSE,
  verbose = FALSE,
  workers = .pmb_v5_default_workers()
) {
  allocation_policy <- match.arg(allocation_policy)
  validate_budget_design(
    n_auctions,
    impression_budget,
    impression_cost
  )

  valid_systems <- c(
    "random_exhaust",
    "random_paced",
    "ridge",
    "two_way",
    "oracle_online",
    "oracle_clairvoyant",
    "thompson",
    "svm",
    "multilevel",
    "neural"
  )
  if (!all(systems %in% valid_systems) || anyDuplicated(systems)) {
    stop("systems contains an unknown or duplicated optimization system.")
  }
  if (!"random_exhaust" %in% systems) {
    stop("systems must include random_exhaust as the primary baseline.")
  }

  for (name in c("n_reps", "workers")) {
    value <- get(name)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value < 1 || value > .Machine$integer.max || value != floor(value)) {
      stop(sprintf("%s must be a positive integer.", name))
    }
  }
  workers <- as.integer(min(workers, n_reps))
  # Explicit seeds below use the caller's RNG algorithms (including sample.kind).
  # Do not use clusterSetRNGStream: that would change V4's statistical draws.
  config_names <- setdiff(names(formals(run_budget_market_comparison)),
                          c("db_path", "workers"))
  config <- mget(config_names, envir = environment())
  config$valid_systems <- valid_systems
  cl <- NULL
  if (workers > 1L) {
    cl <- parallel::makePSOCKcluster(workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterCall(cl, .pmb_v5_initialize_worker,
                          .pmb_v5_source_path, .libPaths(), RNGkind())
  }

  # Initialization and all writes occur in the calling process. Worker errors
  # propagate and on.exit always stops the cluster, including on write failure.
  if (!is.null(db_path)) init_v4_results_db(db_path)
  result_rows <- list()
  event_logs <- list()
  result_index <- 0L
  # Bound transient truth/event payloads to one batch of workers. parLapplyLB
  # returns input order even when jobs complete in a different order.
  for (batch_start in seq.int(1L, n_reps, by = workers)) {
    ids <- seq.int(batch_start, min(n_reps, batch_start + workers - 1L))
    batch <- if (is.null(cl)) {
      lapply(ids, .pmb_v5_replication, config = config)
    } else {
      parallel::parLapplyLB(cl, ids, .pmb_v5_replication, config = config)
    }
    for (j in seq_along(ids)) {
      replication <- ids[j]
      payload <- batch[[j]]
      if (!is.null(db_path)) {
        for (params in payload$truth) {
          write_cell_ground_truth(db_path, replication, params)
        }
      }
      for (row in payload$rows) {
        result_index <- result_index + 1L
        result_rows[[result_index]] <- row
        if (!is.null(db_path)) write_v4_run_summary(db_path, row)
      }
      if (keep_event_log) event_logs <- c(event_logs, payload$events)
      cat(sprintf("Replication %d/%d complete\n", replication, n_reps))
    }
  }

  output <- list(
    results = do.call(rbind, result_rows),
    events = if (keep_event_log) event_logs else NULL,
    design = list(
      n_reps = n_reps,
      n_auctions = n_auctions,
      impression_budget = impression_budget,
      impression_cost = impression_cost,
      burnin_impressions = burnin_impressions,
      refit_interval = refit_interval,
      creative_epsilon = creative_epsilon,
      bid_epsilon = bid_epsilon,
      allocation_policy = allocation_policy,
      thompson_prior_alpha = thompson_prior_alpha,
      thompson_prior_beta = thompson_prior_beta,
      svm_cost = svm_cost,
      neural_hidden_units = neural_hidden_units,
      neural_decay = neural_decay,
      neural_maxit = neural_maxit,
      seed_base = seed_base
    )
  )
  class(output) <- c("pmb_budget_market_comparison", class(output))
  output
}

# =============================================================================
# 6. REPORTING
# =============================================================================

print_budget_market_summary <- function(x) {
  results <- if (inherits(x, "pmb_budget_market_comparison")) {
    x$results
  } else {
    x
  }

  summary_df <- aggregate(
    cbind(
      total_conversions,
      conversion_rate,
      lift_over_random_exhaust,
      budget_utilization,
      cost_per_conversion,
      auction_acceptance_rate
    ) ~ market_structure + system,
    data = results,
    FUN = mean,
    na.rm = TRUE
  )
  summary_df <- summary_df[
    order(summary_df$market_structure, summary_df$system),
    ,
    drop = FALSE
  ]

  cat("\n")
  cat("====================================================================\n")
  cat(" BUDGET-CONSTRAINED FIXED VERSUS CREATIVE-ROTATION CAMPAIGNS\n")
  cat("====================================================================\n")
  print(summary_df, row.names = FALSE, digits = 4)

  cat("\nPaired incremental-conversion tests versus random_exhaust:\n")
  for (structure_name in c("diagonal", "crossed")) {
    structure_rows <- results[
      results$market_structure == structure_name,
      ,
      drop = FALSE
    ]
    baseline <- structure_rows[
      structure_rows$system == "random_exhaust",
      c("replication", "total_conversions"),
      drop = FALSE
    ]
    names(baseline)[2] <- "baseline_conversions"

    for (system_name in intersect(
      c(
        "random_paced", "ridge", "two_way", "thompson", "svm",
        "multilevel", "neural"
      ),
      structure_rows$system
    )) {
      method_rows <- structure_rows[
        structure_rows$system == system_name,
        c("replication", "total_conversions"),
        drop = FALSE
      ]
      names(method_rows)[2] <- "method_conversions"
      paired <- merge(method_rows, baseline, by = "replication")
      difference <- paired$method_conversions -
        paired$baseline_conversions

      cat(sprintf(
        "  %-8s/%-13s: mean incremental conversions = %.2f",
        structure_name,
        system_name,
        mean(difference)
      ))

      if (length(difference) >= 3L && any(difference != 0)) {
        test <- wilcox.test(
          paired$method_conversions,
          paired$baseline_conversions,
          paired = TRUE,
          alternative = "greater",
          exact = FALSE
        )
        cat(sprintf(", paired Wilcoxon p = %.4f\n", test$p.value))
      } else {
        cat(", insufficient nonzero replications for test\n")
      }
    }
  }

  under_spent <- results[
    results$budget_utilization < 0.999,
    c(
      "replication",
      "market_structure",
      "system",
      "budget_utilization"
    ),
    drop = FALSE
  ]
  cat(sprintf(
    "\nRuns ending with material unspent budget: %d of %d\n",
    nrow(under_spent),
    nrow(results)
  ))
  if (nrow(under_spent)) {
    print(under_spent, row.names = FALSE)
  }

  invisible(summary_df)
}

print.pmb_budget_market_comparison <- function(x, ...) {
  print_budget_market_summary(x)
  invisible(x)
}

# =============================================================================
# 7. VALIDATION
# =============================================================================

validate_v4_program <- function(seed = 4321L) {
  set.seed(seed)
  truth <- make_market_truth()
  diagonal <- make_diagonal_params(truth)
  crossed <- make_crossed_params(truth)
  validate_market_pair(diagonal, crossed)

  n_auctions <- 3000L
  impression_budget <- 1000L
  environment <- make_auction_environment(
    truth,
    n_auctions = n_auctions
  )

  systems <- c(
    "random_exhaust",
    "random_paced",
    "ridge",
    "two_way",
    "thompson",
    "svm",
    "multilevel",
    "neural",
    "oracle_online",
    "oracle_clairvoyant"
  )
  validations <- list()

  for (structure in c("diagonal", "crossed")) {
    params <- if (structure == "diagonal") diagonal else crossed
    for (system in systems) {
      set.seed(seed + match(structure, c("diagonal", "crossed")) * 100L +
                 match(system, systems))
      validations[[paste(structure, system, sep = "/")]] <-
        simulate_budget_campaign(
          params = params,
          environment = environment,
          system = system,
          impression_budget = impression_budget,
          burnin_impressions = 300L,
          refit_interval = 300L,
          keep_event_log = TRUE
        )
    }
  }

  stopifnot(
    validations[["diagonal/random_exhaust"]]$impressions_purchased ==
      impression_budget,
    validations[["crossed/random_exhaust"]]$budget_exhausted_at ==
      impression_budget,
    validations[["diagonal/random_paced"]]$impressions_purchased ==
      impression_budget,
    validations[["crossed/random_paced"]]$impressions_purchased ==
      impression_budget,
    validations[["diagonal/oracle_clairvoyant"]]$impressions_purchased ==
      impression_budget,
    validations[["crossed/oracle_clairvoyant"]]$impressions_purchased ==
      impression_budget,
    validations[["diagonal/oracle_clairvoyant"]]$
      cumulative_creative_regret == 0,
    validations[["crossed/oracle_clairvoyant"]]$
      cumulative_creative_regret == 0,
    all(vapply(
      validations,
      function(z) {
        z$impressions_purchased <= impression_budget &&
          z$total_spend <= z$starting_budget &&
          z$remaining_budget >= 0
      },
      logical(1)
    ))
  )

  # In the diagonal market, every purchased impression must use the one
  # creative assigned to that segment, regardless of bidding system.
  diagonal_keys <- validations[["diagonal/ridge"]]$events
  bought <- diagonal_keys$bid
  expected_key <- make_key(
    diagonal_keys$segment[bought],
    paste0(
      "creative",
      sub("seg", "", diagonal_keys$segment[bought])
    )
  )
  stopifnot(identical(diagonal_keys$key[bought], expected_key))

  # Neural learner must refit and provide usable predictions in both markets.
  for (structure in c("diagonal", "crossed")) {
    neural_result <- validations[[paste(structure, "neural", sep = "/")]]
    stopifnot(
      neural_result$market_structure == structure,
      any(is.finite(
        neural_result$events$predicted[neural_result$events$bid]
      ))
    )
  }

  # Proportional Thompson must log a defined selection probability for every
  # purchased crossed-market impression; coordinated Thompson logged NA here.
  thompson_events <- validations[["crossed/thompson"]]$events
  thompson_bought <- thompson_events$bid
  stopifnot(
    all(is.finite(
      thompson_events$creative_selection_probability[thompson_bought]
    )),
    validations[["crossed/thompson"]]$allocation_policy == "proportional"
  )

  cat("V4 validation passed.\n")
  invisible(TRUE)
}

# Example:
# source("C:/R_Code/pmb_market_sim_v5.R")
# validate_v4_program()
#
# study_v5 <- run_budget_market_comparison(
#   n_reps = 60,
#   n_auctions = 500000,
#   impression_budget = 155000,
#   impression_cost = 1,
#   burnin_impressions = 10000,
#   refit_interval = 5000,
#   creative_epsilon = 0.10,
#   bid_epsilon = 0.05,
#   allocation_policy = "proportional",
#   workers = 8L, # Use 1L for serial execution.
#   db_path = "C:/R_Code/pmb_market_v5.db"
# )
# print(study_v5)
#
#then save
#saveRDS(
#  study_v4,
#  "C:/R_Code/study_v4_n60_nn.rds"
#)

# Small exact reproducibility check: includes stochastic learning, refits,
# both structures, and full event logs. No SQLite files are needed.
validate_v5_reproducibility <- function() {
  args <- list(
    n_reps = 2L, n_auctions = 600L, impression_budget = 250L,
    burnin_impressions = 100L, refit_interval = 100L,
    base_rate = 0.15, seed_base = 4321L, keep_event_log = TRUE,
    neural_maxit = 50L
  )
  serial <- do.call(run_budget_market_comparison, c(args, list(workers = 1L)))
  parallel <- do.call(run_budget_market_comparison, c(args, list(workers = 2L)))
  stopifnot(identical(serial, parallel))
  cat("V5 serial/PSOCK reproducibility validation passed (exact equality).\n")
  invisible(TRUE)
}
# validate_v5_reproducibility()

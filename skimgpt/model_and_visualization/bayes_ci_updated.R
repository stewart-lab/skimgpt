# an LLM is asked to score two mutually exclusive hypotheses to explain some
# phenomenon, using literature evidence retrieved from pubmed. the LLM
# produces a score between 0 and 100, with 0 favoring one hypothesis and 100
# favoring the other hypothesis. a score of 50 means that the LLM thinks the
# two hypotheses are equally likely based on the provided texts.

# the goal of this system is not to say if hyp1 or hyp2 are true. the goal
# is to estimate when the literature came to a consensus regarding hyp1
# or hyp2.

# the LLM score is susceptible to drawing strong conclusions from few
# abstracts. so, we introduce a Bayesian prior here to shrink the posterior
# score towards 50. if there is little evidence (i.e., few abstracts), then
# the prior dominates. if there is a lot of evidence, then the prior doesn't
# influence the score much.

# each LLM call gets up to 50 abstracts. the pool to draw abstracts from can
# be very large (thousands of abstracts). if the literature is sparse, the
# same abstracts will be sent to the LLM repeatedly, so only unique PMIDs are
# counted as evidence.

# implementation:
# - n_effective is the effective number of abstracts observed
# - we view the LLM score as a noisy approximation of the 'true LLM score'.
#   we model sources of noise (variance between LLM calls and
#   literature sampling) that we add together.
# - we weight an LLM observation by the n_effective number of abstracts.
#   this is the 'learning rate'.
# - abstracts are not treated as independent evidence. papers in a field
#   cite each other and share priors.

# notes:
# - the HDI (error bar) represents the pipeline's (i.e., our) ability to
#   estimate how confident the literature was. it does not represent how
#   confident the literature was.
# - a high error bar is generally caused by low numbers of abstracts. we don't
#   have enough information to get a solid read on what the scientific opinion
#   was at that point in time.
# - how confident the literature was at a given point in time is represented
#   by the LLM score.
# - sampling a small portion of the available literature is not penalized
#   (i.e., sampling 100 out of 10000 abstracts is treated the same as sampling
#   100 out of 100 abstracts).
# - "how confident the literature is" is a vague quantity without a
#   statistical definition. but, imagine we have a field of 100 experts, each
#   publishing once per year, and they support hyp1 and hyp2 in equal measure
#   (i.e., 50 abstracts each). we would say the field is probably evenly
#   divided (minimum "literature confidence"). because we have 100 abstracts
#   to sample from, we can say that our ability to measure confidence is
#   pretty high - we are confident in our ability to measure literature
#   confidence.

# NOTE ON PACKAGES: this script ports skimgpt/bayes_ci_updated.py, which uses
# a closed-form Beta posterior (no MLE/MoM fitting, no per-year diagnostic
# density plots). That means bayestestR, EnvStats, schoolmath, zoo, viridis,
# and gridExtra - all used by the older bayes_citest.R model - aren't needed
# here; qbeta/dbeta/optimize (base R) replace them exactly. jsonlite is added
# because the input is now per-iteration JSON files, not a single CSV.

library(optparse)
library(jsonlite)
library(dplyr)
library(ggplot2)
library(patchwork)

# ---------------------------------------------------------------------------
# Beta-posterior helpers (mirrors bayes_ci_updated.py)
# ---------------------------------------------------------------------------

A0 <- 1.2 # strength of the prior. pseudo-abstracts per side
RHO <- 0 # intra-field correlation
SIGMA2_CALL <- 0.0007823878 # variance of a single LLM call, 0-1 scale, calculated by rerunning the same abstracts through 50 iterations

n_effective <- function(m, n_calls, rho = RHO, sigma2_call = SIGMA2_CALL, theta = 0.5) {
  if (m == 0 || n_calls == 0) {
    return(0.0)
  }
  n_lit <- m / (1.0 + (m - 1) * rho)
  n_call <- (theta * (1 - theta) * n_calls) / sigma2_call
  1.0 / (1.0 / n_lit + 1.0 / n_call)
}

posterior_params <- function(calls, a0 = A0) {
  scores <- vapply(calls, function(cc) cc$score / 100.0, numeric(1))
  weights <- vapply(calls, function(cc) nrow(cc$pmids), numeric(1))

  if (sum(weights) == 0) {
    s_bar <- 0.5
  } else {
    s_bar <- stats::weighted.mean(scores, weights)
  }
  s_bar <- min(max(s_bar, 1e-6), 1 - 1e-6)

  all_pmids <- unique(unlist(lapply(calls, function(cc) cc$pmids$pmid)))
  m <- length(all_pmids)
  n_eff <- n_effective(m, length(calls), theta = s_bar)

  a <- a0 + n_eff * s_bar
  b <- a0 + n_eff * (1.0 - s_bar)

  list(a = a, b = b)
}

# exact Highest Density Interval for Beta(a,b), matching the Python
# implementation's use of scipy.optimize.minimize_scalar over the analytic CDF
# (rather than bayestestR's sample-based HDI, which would add Monte Carlo noise).
hdi_beta <- function(a, b, level = 0.95) {
  if (abs(a - b) < 1e-9) {
    tail <- (1 - level) / 2
    return(c(stats::qbeta(tail, a, b), stats::qbeta(1 - tail, a, b)))
  }

  width <- function(lo_p) {
    stats::qbeta(lo_p + level, a, b) - stats::qbeta(lo_p, a, b)
  }

  res <- stats::optimize(width, interval = c(1e-9, 1 - level - 1e-9))
  lo_p <- res$minimum
  c(stats::qbeta(lo_p, a, b), stats::qbeta(lo_p + level, a, b))
}

posterior_beta <- function(calls, n_theta = 300) {
  params <- posterior_params(calls)
  grid <- seq(0.005, 0.995, length.out = n_theta)
  p <- stats::dbeta(grid, params$a, params$b)
  list(grid = grid, p = p / sum(p), a = params$a, b = params$b)
}

# ---------------------------------------------------------------------------
# Per-year summary table underlying both plots
# ---------------------------------------------------------------------------

timecourse_data <- function(data, level = 0.95, hyp1_label = "hyp1", hyp2_label = "hyp2") {
  years <- sort(as.numeric(names(data)))

  rows <- lapply(years, function(year) {
    calls <- data[[as.character(year)]]
    pb <- posterior_beta(calls)

    hdi <- hdi_beta(pb$a, pb$b, level) * 100
    posterior_mode <- pb$grid[which.max(pb$p)] * 100

    scores <- vapply(calls, function(cc) cc$score, numeric(1))
    llm_mean <- mean(scores)

    all_pmids <- unique(unlist(lapply(calls, function(cc) cc$pmids$pmid)))
    total_unique_abstracts <- length(all_pmids)

    per_iter <- lapply(calls, function(cc) {
      n_h1 <- sum(cc$pmids$label == "supports_H1")
      n_h2 <- sum(cc$pmids$label == "supports_H2")
      n_both <- sum(cc$pmids$label == "both")
      adj_h1 <- n_h1 + n_both / 2
      adj_h2 <- n_h2 + n_both / 2
      denom <- adj_h1 + adj_h2
      proportion <- if (denom == 0) NA_real_ else adj_h1 / denom
      c(n_h1 = n_h1, n_h2 = n_h2, n_both = n_both, proportion = proportion)
    })
    per_iter_mat <- do.call(rbind, per_iter)

    avg_supports_H1 <- mean(per_iter_mat[, "n_h1"])
    avg_supports_H2 <- mean(per_iter_mat[, "n_h2"])
    avg_both <- mean(per_iter_mat[, "n_both"])
    avg_abstracts_per_iteration <- mean(rowSums(per_iter_mat[, c("n_h1", "n_h2", "n_both"), drop = FALSE]))
    avg_proportion <- mean(per_iter_mat[, "proportion"], na.rm = TRUE)

    data.frame(
      year = year, hyp1 = hyp1_label, hyp2 = hyp2_label,
      mean_llm_score = llm_mean, posterior_score = posterior_mode,
      hdi_level = level, hdi_lo = hdi[1], hdi_hi = hdi[2],
      total_unique_abstracts = total_unique_abstracts,
      avg_abstracts_per_iteration = avg_abstracts_per_iteration,
      avg_supports_H1 = avg_supports_H1, avg_supports_H2 = avg_supports_H2, avg_both = avg_both,
      avg_proportion = avg_proportion
    )
  })

  do.call(rbind, rows)
}

write_timecourse_csv <- function(data, path, level = 0.95, hyp1_label = "hyp1", hyp2_label = "hyp2") {
  df <- timecourse_data(data, level = level, hyp1_label = hyp1_label, hyp2_label = hyp2_label)
  write.csv(df, file = path, row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Unique-abstract counts per year, split by label (winning label per PMID,
# ties broken supports_H1 > supports_H2 > both)
# ---------------------------------------------------------------------------

TIE_BREAK_ORDER <- c("supports_H1", "supports_H2", "both")

label_counts_data <- function(data) {
  years <- sort(as.numeric(names(data)))

  rows <- lapply(years, function(year) {
    calls <- data[[as.character(year)]]
    pmid_rows <- do.call(rbind, lapply(calls, function(cc) cc$pmids))

    if (is.null(pmid_rows) || nrow(pmid_rows) == 0) {
      return(data.frame(year = year, label = TIE_BREAK_ORDER, count = 0))
    }

    best_label <- pmid_rows %>%
      dplyr::count(pmid, label, name = "n") %>%
      dplyr::mutate(rank = match(label, TIE_BREAK_ORDER)) %>%
      dplyr::arrange(pmid, dplyr::desc(n), rank) %>%
      dplyr::group_by(pmid) %>%
      dplyr::slice(1) %>%
      dplyr::ungroup()

    tab <- table(factor(best_label$label, levels = TIE_BREAK_ORDER))
    data.frame(year = year, label = TIE_BREAK_ORDER, count = as.numeric(tab))
  })

  do.call(rbind, rows)
}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

add_milestones <- function(p, proposed_date = NA, decision_date = NA, decision_label = NA,
                           reconsidered_date = NA, show_labels = TRUE, label_y = 5) {
  milestones <- list()
  if (!is.na(proposed_date)) {
    milestones[[length(milestones) + 1]] <- list(date = proposed_date, color = "darkred", label = "proposed")
  }
  if (!is.na(decision_date)) {
    if (!decision_label %in% c("accepted", "rejected", "unknown")) {
      stop('decision_label must be "accepted" or "rejected" or "unknwon"')
    }
    milestones[[length(milestones) + 1]] <- list(date = decision_date, color = "black", label = decision_label)
  }
  if (!is.na(reconsidered_date)) {
    milestones[[length(milestones) + 1]] <- list(date = reconsidered_date, color = "dimgray", label = "reconsidered")
  }

  for (m in milestones) {
    p <- p + geom_vline(xintercept = m$date, linetype = "dashed", color = m$color, linewidth = 0.6)
    if (show_labels) {
      p <- p + annotate("text",
        x = m$date, y = label_y, label = m$label,
        angle = 90, hjust = 0, vjust = -0.3, size = 2.6, color = m$color
      )
    }
  }

  p
}

plot_timecourse_panel <- function(df_tc, level = 0.95, title = NULL,
                                  hyp1_label = "hyp1", hyp2_label = "hyp2", dot_size = 1.8,
                                  proposed_date = NA, decision_date = NA, decision_label = NA,
                                  reconsidered_date = NA, show_milestone_labels = TRUE) {
  hdi_name <- sprintf("%.0f%% HDI", level * 100)

  p <- ggplot(df_tc, aes(x = year)) +
    geom_ribbon(aes(ymin = hdi_lo, ymax = hdi_hi, fill = hdi_name), alpha = 0.35) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_line(aes(y = posterior_score, color = "posterior"), linewidth = 0.9) +
    geom_point(aes(y = mean_llm_score, color = "LLM score"), size = dot_size) +
    scale_fill_manual(name = NULL, values = setNames("grey50", hdi_name)) +
    scale_color_manual(name = NULL, values = c("posterior" = "darkblue", "LLM score" = "black")) +
    annotate("text",
      x = 1975, y = 97, label = paste("favors", hyp1_label),
      hjust = 0, vjust = 1, size = 3, color = "gray40"
    ) +
    annotate("text",
      x = 1975, y = 3, label = paste("favors", hyp2_label),
      hjust = 0, vjust = 0, size = 3, color = "gray40"
    ) +
    scale_x_continuous(breaks = seq(1975, 2025, 1)) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, 10)) +
    coord_cartesian(xlim = c(1975, 2025)) +
    labs(title = title %||% "Literature support over time", x = NULL, y = "score") +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black"),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "inside",
      legend.position.inside = c(0.98, 0.02),
      legend.justification = c("right", "bottom"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 8),
      plot.margin = margin(t = 5, r = 5, b = 0, l = 5)
    )

  add_milestones(p, proposed_date, decision_date, decision_label, reconsidered_date,
    show_labels = show_milestone_labels
  )
}

plot_label_counts_panel <- function(df_counts, hyp1_label = "hyp1", hyp2_label = "hyp2",
                                    normalize = FALSE,
                                    proposed_date = NA, decision_date = NA, decision_label = NA,
                                    reconsidered_date = NA, show_milestone_labels = FALSE) {
  label_display <- c(supports_H1 = hyp1_label, supports_H2 = hyp2_label, both = "both")
  label_colors <- c(supports_H1 = "darkorange", supports_H2 = "purple", both = "green")

  df_counts <- df_counts %>%
    dplyr::group_by(year) %>%
    dplyr::mutate(pct = if (sum(count) > 0) 100 * count / sum(count) else 0) %>%
    dplyr::ungroup()
  df_counts$label <- factor(df_counts$label, levels = TIE_BREAK_ORDER) # H1, H2, both -> H1 on top of stack

  y_col <- if (normalize) "pct" else "count"

  p <- ggplot(df_counts, aes(x = year, y = .data[[y_col]], fill = label)) +
    geom_col(width = 0.6) +
    scale_fill_manual(name = "label", values = label_colors, breaks = TIE_BREAK_ORDER, labels = label_display) +
    scale_x_continuous(breaks = seq(1975, 2025, 1)) +
    coord_cartesian(xlim = c(1975, 2025)) +
    labs(x = "year", y = if (normalize) "% of abstracts" else "unique abstracts") +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black"),
      axis.text.x = element_text(angle = 90, size = 7, vjust = 0.5),
      legend.position = "inside",
      legend.position.inside = c(0.02, 0.98),
      legend.justification = c("left", "top"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 8),
      plot.margin = margin(t = 0, r = 5, b = 5, l = 5)
    )

  if (normalize) p <- p + scale_y_continuous(limits = c(0, 100))

  add_milestones(p, proposed_date, decision_date, decision_label, reconsidered_date,
    show_labels = show_milestone_labels
  )
}

plot_llm_vs_proportion_panel <- function(df_tc, title = NULL,
                                         proposed_date = NA, decision_date = NA, decision_label = NA,
                                         reconsidered_date = NA, show_milestone_labels = TRUE) {
  p <- ggplot(df_tc, aes(x = year)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "black", linewidth = 0.5) +
    geom_line(aes(y = mean_llm_score / 100, color = "LLM score"), linewidth = 0.9) +
    geom_point(aes(y = mean_llm_score / 100, color = "LLM score"), size = 1.8) +
    geom_line(aes(y = avg_proportion, color = "abstract proportion"), linewidth = 0.9) +
    geom_point(aes(y = avg_proportion, color = "abstract proportion"), size = 1.8) +
    scale_color_manual(name = NULL, values = c("LLM score" = "darkblue", "abstract proportion" = "darkorange")) +
    scale_x_continuous(breaks = seq(1975, 2025, 1)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
    coord_cartesian(xlim = c(1975, 2025)) +
    labs(title = title %||% "LLM score vs. abstract support proportion", x = "year", y = "value (0-1)") +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black"),
      axis.text.x = element_text(angle = 90, size = 7, vjust = 0.5),
      legend.position = "inside",
      legend.position.inside = c(0.98, 0.02),
      legend.justification = c("right", "bottom"),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 8),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )

  add_milestones(p, proposed_date, decision_date, decision_label, reconsidered_date,
    show_labels = show_milestone_labels, label_y = 0.05
  )
}

plot_combined <- function(data, hyp1_label = "hyp1", hyp2_label = "hyp2", level = 0.95,
                          title = NULL, normalize = FALSE, dot_size = 1.8,
                          proposed_date = NA, decision_date = NA, decision_label = NA,
                          reconsidered_date = NA) {
  df_tc <- timecourse_data(data, level = level, hyp1_label = hyp1_label, hyp2_label = hyp2_label)
  df_counts <- label_counts_data(data)

  p_top <- plot_timecourse_panel(df_tc,
    level = level, title = title,
    hyp1_label = hyp1_label, hyp2_label = hyp2_label, dot_size = dot_size,
    proposed_date = proposed_date, decision_date = decision_date,
    decision_label = decision_label, reconsidered_date = reconsidered_date,
    show_milestone_labels = TRUE
  )

  p_bottom <- plot_label_counts_panel(df_counts,
    hyp1_label = hyp1_label, hyp2_label = hyp2_label,
    normalize = normalize,
    proposed_date = proposed_date, decision_date = decision_date,
    decision_label = decision_label, reconsidered_date = reconsidered_date,
    show_milestone_labels = FALSE
  )

  p_top / p_bottom + patchwork::plot_layout(heights = c(5, 3))
}

`%||%` <- function(a, b) if (is.null(a) || is.na(a)) b else a

# ---------------------------------------------------------------------------
# Data loading: walk data_dir for gpt_direct_comp.json result files, matching
# each to the nearest ancestor config.json for its censor year
# ---------------------------------------------------------------------------

get_data <- function(data_dir) {
  if (!dir.exists(data_dir)) {
    stop(paste0("Directory does not exist: ", data_dir))
  }

  all_files <- list.files(data_dir, recursive = TRUE, full.names = TRUE)
  json_files <- all_files[grepl("gpt_direct_comp\\.json$", all_files) & !grepl("\\.backup", all_files)]

  data <- list()

  for (iteration_json in json_files) {
    config_dir <- dirname(iteration_json)
    config_json <- NULL
    repeat {
      candidate <- file.path(config_dir, "config.json")
      if (file.exists(candidate)) {
        config_json <- candidate
        break
      }
      parent <- dirname(config_dir)
      if (parent == config_dir) break # reached filesystem root without finding one
      config_dir <- parent
    }
    if (is.null(config_json)) {
      stop(paste0("No config.json found for ", iteration_json))
    }

    config_content <- jsonlite::fromJSON(config_json, simplifyVector = FALSE)
    km <- config_content$JOB_SPECIFIC_SETTINGS$km_with_gpt
    year <- km$censor_year_upper
    if (is.null(year)) {
      year <- km$km_with_gpt$censor_year_upper
    }

    iter_content <- jsonlite::fromJSON(iteration_json, simplifyVector = FALSE)
    hyp_eval <- iter_content[[1]]$Hypothesis_Comparison$Result

    if (length(hyp_eval) == 0) {
      message(paste0("no result for ", iteration_json))
      next
    }

    hyp_eval <- hyp_eval[[1]]
    keep_labels <- c("supports_H1", "supports_H2", "both")
    per_abstract <- Filter(function(a) a$label %in% keep_labels, hyp_eval$per_abstract)

    if (length(per_abstract) == 0) {
      pmids_df <- data.frame(pmid = character(0), label = character(0), stringsAsFactors = FALSE)
    } else {
      pmids_df <- data.frame(
        pmid = vapply(per_abstract, function(a) as.character(a$pmid), character(1)),
        label = vapply(per_abstract, function(a) a$label, character(1)),
        stringsAsFactors = FALSE
      )
    }
    llm_score <- hyp_eval$score

    year_key <- as.character(year)
    if (is.null(data[[year_key]])) data[[year_key]] <- list()
    data[[year_key]][[length(data[[year_key]]) + 1]] <- list(score = llm_score, pmids = pmids_df)
  }

  data
}

# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

option_list <- list(
  make_option(c("-d", "--data_dir"),
    type = "character", default = NULL,
    help = "directory holding gpt_direct_comp.json result files and config.json; the output_model_ folder is created here",
    metavar = "character"
  ),
  make_option(c("--hyp1_label"),
    type = "character", default = "genetic predisposition",
    help = "display term for hypothesis 1 (H1) [default %default]", metavar = "character"
  ),
  make_option(c("--hyp2_label"),
    type = "character", default = "vaccines",
    help = "display term for hypothesis 2 (H2) [default %default]", metavar = "character"
  ),
  make_option(c("--level"),
    type = "double", default = 0.95,
    help = "HDI credible level [default %default]", metavar = "double"
  ),
  make_option(c("--dot_size"),
    type = "double", default = 1.8,
    help = "marker size for the LLM score dots [default %default]", metavar = "double"
  ),
  make_option(c("-t", "--title"),
    type = "character", default = NULL,
    help = "title for the timecourse panel", metavar = "character"
  ),
  make_option(c("--normalize"),
    action = "store_true", default = FALSE,
    help = "normalize the label-counts bars to percent of abstracts"
  ),
  make_option(c("--proposed_date"),
    type = "integer", default = 1998,
    help = "year the hypothesis was proposed [default %default]", metavar = "integer"
  ),
  make_option(c("--decision_date"),
    type = "integer", default = 2011,
    help = "year the literature accepted or rejected the hypothesis [default %default]", metavar = "integer"
  ),
  make_option(c("--decision_label"),
    type = "character", default = "accepted",
    help = '"accepted" or "rejected"; required if decision_date is set [default %default]', metavar = "character"
  ),
  make_option(c("--reconsidered_date"),
    type = "integer", default = NA,
    help = "year the hypothesis was reconsidered, if applicable", metavar = "integer"
  )
)

# example run
# Rscript skimgpt/bayes_ci_updated.R --data_dir /w5home/bmoore/km_skim_stats/Supplemental_data2/historical_controversial_hyps_1975-2025/Autism_GeneticPredispositionVsVaccines/ --decision_date 2011 --decision_label rejected

main <- function() {
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)

  if (is.null(opt$data_dir)) {
    print_help(opt_parser)
    stop("--data_dir is required", call. = FALSE)
  }

  data_dir <- normalizePath(opt$data_dir)
  timestamp <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  output_dir <- file.path(data_dir, paste0("output_model_", timestamp))
  dir.create(output_dir, mode = "0777", showWarnings = FALSE, recursive = TRUE)

  data <- get_data(data_dir)

  combined_plot <- plot_combined(
    data,
    hyp1_label = opt$hyp1_label, hyp2_label = opt$hyp2_label, level = opt$level,
    title = opt$title, normalize = opt$normalize, dot_size = opt$dot_size,
    proposed_date = opt$proposed_date, decision_date = opt$decision_date,
    decision_label = opt$decision_label, reconsidered_date = opt$reconsidered_date
  )
  ggsave(file.path(output_dir, "combined.pdf"), combined_plot, width = 8, height = 8, units = "in")

  df_tc <- timecourse_data(data, level = opt$level, hyp1_label = opt$hyp1_label, hyp2_label = opt$hyp2_label)
  write.csv(df_tc, file = file.path(output_dir, "timecourse_data.csv"), row.names = FALSE)

  llm_vs_proportion_plot <- plot_llm_vs_proportion_panel(df_tc,
    title = opt$title,
    proposed_date = opt$proposed_date, decision_date = opt$decision_date,
    decision_label = opt$decision_label, reconsidered_date = opt$reconsidered_date
  )
  ggsave(file.path(output_dir, "llm_vs_proportion.pdf"), llm_vs_proportion_plot, width = 8, height = 5, units = "in")

  # reproducibility: record the exact args and package/session state for this run
  parameter_df <- data.frame(
    Parameter = c(
      "data_dir", "hyp1_label", "hyp2_label", "level", "dot_size", "title",
      "normalize", "proposed_date", "decision_date", "decision_label", "reconsidered_date",
      "A0", "rho", "sigma2_call"
    ),
    Value = c(
      data_dir, opt$hyp1_label, opt$hyp2_label, opt$level, opt$dot_size,
      opt$title %||% "", opt$normalize, opt$proposed_date, opt$decision_date,
      opt$decision_label, opt$reconsidered_date, A0, RHO, SIGMA2_CALL
    )
  )
  write.csv(parameter_df, file = file.path(output_dir, "parameters.csv"), row.names = FALSE)
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))

  cat("Plots and data saved to", output_dir, "\n")
}

if (sys.nframe() == 0) {
  main()
}

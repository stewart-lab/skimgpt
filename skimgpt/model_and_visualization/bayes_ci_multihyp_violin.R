# Compares many DIFFERENT hypothesis-pairs (each its own H1 vs H2, e.g. one
# per disease/topic) at a single timepoint, using the same closed-form Beta
# posterior model as bayes_ci_updated.R (n_effective shrinkage + exact HDI -
# see that file for the full model rationale). Where bayes_ci_updated.R walks
# one hypothesis-pair across many censor years to show a timecourse, this
# script walks many hypothesis-pair directories at one censor-year window and
# shows them side by side as a violin plot, ordered by posterior mean.
#
# Expected input: projpath contains one subdirectory per DCH run (e.g.
# "output_<timestamp>_<topic>_kmgptdch_<years>_<model>/", the standard
# SKiM-GPT DCH run-output naming), each holding results/iteration_N/
# *_km_with_gpt_direct_comp.json files. A subdirectory may itself contain
# several distinct hypothesis-pairs when it comes from an A-term-list run
# (one *_direct_comp.json basename per A term, e.g. one per gene, all
# comparing that gene against the same fixed B-term pair) - each such
# basename group is pooled across its own iterations into its own posterior
# and gets its own violin (no year-splitting within a group - this is a
# snapshot, not a timecourse). H1/H2 short labels are derived automatically
# from the JSON's "hypothesis1"/"hypothesis2" text by diffing out the shared
# wording they're templated from (see short_hypothesis_labels()); when a
# subdirectory holds multiple hypothesis-pairs, the varying term across
# their "hypothesis1" texts (e.g. the A term) is extracted the same way (see
# extract_varying_term()) and appended to the subdirectory-derived topic
# label so each pair gets its own axis entry instead of being pooled
# together.
#
# Packages: kept from the user's older multi-hypothesis violin script -
# ggplot2, patchwork, dplyr, optparse, viridis (used here for the violin
# fill scale, unlike bayes_ci_updated.R where it wasn't needed) - plus
# jsonlite for the JSON inputs. bayestestR/EnvStats/schoolmath/zoo/gridExtra
# are dropped for the same reason as bayes_ci_updated.R: no MLE/MoM fitting
# or per-hypothesis diagnostic density plots here either.

library(optparse)
library(jsonlite)
library(dplyr)
library(ggplot2)
library(viridis)
library(patchwork)

# reuse the exact Beta-posterior model (n_effective, posterior_params, hdi_beta,
# A0/RHO/SIGMA2_CALL) from bayes_ci_updated.R so both scripts always agree.
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  script_path <- sub(file_flag, "", args[grep(file_flag, args)])
  if (length(script_path) == 0) {
    return(getwd())
  }
  normalizePath(dirname(script_path[1]))
}
source(file.path(get_script_dir(), "bayes_ci_updated.R"))

# ---------------------------------------------------------------------------
# Per-directory data loading: a directory can hold more than one distinct
# hypothesis-pair (one *_direct_comp.json basename per pair - e.g. one per A
# term, under an A-term-list DCH run). Group files by basename and pool each
# group's iterations into its own list of "calls", pulling hypothesis1/
# hypothesis2 straight from the JSON (same for every iteration within a
# group, so the first file in the group is enough). Returns one list entry
# per distinct hypothesis-pair found.
# ---------------------------------------------------------------------------

load_hypothesis_groups <- function(topic_dir) {
  files <- list.files(topic_dir, pattern = "gpt_direct_comp\\.json$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("\\.backup", files)]
  if (length(files) == 0) {
    return(NULL)
  }

  keep_labels <- c("supports_H1", "supports_H2", "both")

  groups <- lapply(split(files, basename(files)), function(group_files) {
    calls <- list()
    hyp1_text <- NULL
    hyp2_text <- NULL

    for (f in group_files) {
      content <- jsonlite::fromJSON(f, simplifyVector = FALSE)
      hc <- content[[1]]$Hypothesis_Comparison
      if (is.null(hyp1_text)) {
        hyp1_text <- hc$hypothesis1
        hyp2_text <- hc$hypothesis2
      }

      result <- hc$Result[[1]]
      per_abstract <- Filter(function(a) a$label %in% keep_labels, result$per_abstract)

      if (length(per_abstract) == 0) {
        pmids_df <- data.frame(pmid = character(0), label = character(0), stringsAsFactors = FALSE)
      } else {
        pmids_df <- data.frame(
          pmid = vapply(per_abstract, function(a) as.character(a$pmid), character(1)),
          label = vapply(per_abstract, function(a) a$label, character(1)),
          stringsAsFactors = FALSE
        )
      }

      calls[[length(calls) + 1]] <- list(score = result$score, pmids = pmids_df)
    }

    list(calls = calls, hypothesis1 = hyp1_text, hypothesis2 = hyp2_text)
  })

  unname(groups)
}

# ---------------------------------------------------------------------------
# Derive short H1/H2 labels by diffing out the wording hypothesis1/hypothesis2
# share (they're both instantiations of the same template, differing only in
# the term that was substituted in) - e.g.
#   "The main cause of Schizophrenia is due to dopamine signaling."
#   "The main cause of Schizophrenia is due to glutamate signaling."
# -> "dopamine" / "glutamate"
# Falls back to the full text if no common prefix/suffix is found.
# ---------------------------------------------------------------------------

short_hypothesis_labels <- function(h1, h2) {
  clean <- function(words) trimws(gsub("[.,;:]+$", "", paste(words, collapse = " ")))

  w1 <- strsplit(trimws(h1), "\\s+")[[1]]
  w2 <- strsplit(trimws(h2), "\\s+")[[1]]
  max_common <- min(length(w1), length(w2))

  n_pre <- 0
  while (n_pre < max_common && w1[n_pre + 1] == w2[n_pre + 1]) n_pre <- n_pre + 1

  n_suf <- 0
  max_suf <- max_common - n_pre
  while (n_suf < max_suf && w1[length(w1) - n_suf] == w2[length(w2) - n_suf]) n_suf <- n_suf + 1

  mid1 <- if (n_pre + n_suf < length(w1)) w1[(n_pre + 1):(length(w1) - n_suf)] else character(0)
  mid2 <- if (n_pre + n_suf < length(w2)) w2[(n_pre + 1):(length(w2) - n_suf)] else character(0)

  term1 <- if (length(mid1) > 0) clean(mid1) else clean(w1)
  term2 <- if (length(mid2) > 0) clean(mid2) else clean(w2)

  list(term1 = term1, term2 = term2)
}

# ---------------------------------------------------------------------------
# N-way generalization of short_hypothesis_labels()'s diff: given several
# strings templated from the same wording but with one term substituted
# (e.g. one hypothesis1 per A term, all sharing the same B-term comparison),
# diff out the prefix/suffix common to ALL of them and return each string's
# differing middle segment (the substituted term). Falls back to the full
# string for any input where no common prefix/suffix could be established.
# ---------------------------------------------------------------------------

extract_varying_term <- function(strings) {
  clean <- function(words) trimws(gsub("[.,;:]+$", "", paste(words, collapse = " ")))

  word_lists <- lapply(strings, function(s) strsplit(trimws(s), "\\s+")[[1]])
  lens <- vapply(word_lists, length, integer(1))
  max_common <- if (length(lens) > 0) min(lens) else 0

  n_pre <- 0
  while (n_pre < max_common) {
    words_at_pos <- vapply(word_lists, function(w) w[n_pre + 1], character(1))
    if (length(unique(words_at_pos)) != 1) break
    n_pre <- n_pre + 1
  }

  n_suf <- 0
  max_suf <- max_common - n_pre
  while (n_suf < max_suf) {
    words_at_pos <- vapply(word_lists, function(w) w[length(w) - n_suf], character(1))
    if (length(unique(words_at_pos)) != 1) break
    n_suf <- n_suf + 1
  }

  vapply(word_lists, function(w) {
    mid <- if (n_pre + n_suf < length(w)) w[(n_pre + 1):(length(w) - n_suf)] else character(0)
    if (length(mid) > 0) clean(mid) else clean(w)
  }, character(1))
}

topic_from_dirname <- function(dir_name) {
  m <- regmatches(dir_name, regexec("^output_[0-9]+_(.+?)_kmgptdch", dir_name))[[1]]
  if (length(m) >= 2) m[2] else dir_name
}

# ---------------------------------------------------------------------------
# Per-topic posterior summary (exact Beta(a,b), same math as bayes_ci_updated.R)
# ---------------------------------------------------------------------------

summarize_topic <- function(topic, hyp1_label, hyp2_label, hypothesis1, hypothesis2,
                            calls, level = 0.95, n_samples = 2000) {
  params <- posterior_params(calls)
  a <- params$a
  b <- params$b

  hdi <- hdi_beta(a, b, level) * 100
  posterior_mean <- (a / (a + b)) * 100
  scores <- vapply(calls, function(cc) cc$score, numeric(1))

  all_pmids <- unique(unlist(lapply(calls, function(cc) cc$pmids$pmid)))

  # Mean per-iteration count of abstracts labeled as supporting H1/H2 -
  # the raw evidence tally that accompanies the posterior estimate in the
  # side-by-side bar plot. Averaged (not summed) across calls/iterations so
  # topics are comparable regardless of how many iterations each ran.
  count_label <- function(lbl) vapply(calls, function(cc) sum(cc$pmids$label == lbl), numeric(1))
  mean_support_h1 <- mean(count_label("supports_H1"))
  mean_support_h2 <- mean(count_label("supports_H2"))

  list(
    summary = data.frame(
      topic = topic, hyp1_label = hyp1_label, hyp2_label = hyp2_label,
      hypothesis1 = hypothesis1, hypothesis2 = hypothesis2,
      n_calls = length(calls), n_unique_pmids = length(all_pmids),
      mean_llm_score = mean(scores), posterior_mean = posterior_mean,
      hdi_level = level, hdi_lo = hdi[1], hdi_hi = hdi[2],
      shape1 = a, shape2 = b,
      mean_support_h1 = mean_support_h1, mean_support_h2 = mean_support_h2
    ),
    # samples for the violin shape: drawn directly from the exact posterior
    # Beta(a,b) - not a resampling/refitting step, just visualizing that
    # closed-form distribution.
    samples = data.frame(topic = topic, posterior = stats::rbeta(n_samples, a, b) * 100)
  )
}

# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

option_list <- list(
  make_option(c("-p", "--projpath"),
    type = "character", default = NULL,
    help = "directory containing one subdirectory per hypothesis-pair run",
    metavar = "character"
  ),
  make_option(c("--level"),
    type = "double", default = 0.95,
    help = "HDI credible level [default %default]", metavar = "double"
  ),
  make_option(c("--n_samples"),
    type = "integer", default = 2000,
    help = "posterior draws per hypothesis-pair used for the violin shape [default %default]",
    metavar = "integer"
  )
)

main <- function() {
  opt_parser <- OptionParser(option_list = option_list)
  opt <- parse_args(opt_parser)

  if (is.null(opt$projpath)) {
    print_help(opt_parser)
    stop("--projpath is required", call. = FALSE)
  }

  proj_path <- normalizePath(opt$projpath)
  timestamp <- format(Sys.time(), "%Y_%m_%d_%H_%M_%S")
  output_dir <- file.path(proj_path, paste0("output_model_", timestamp))
  dir.create(output_dir, mode = "0777", showWarnings = FALSE, recursive = TRUE)

  # --projpath can point either at a parent folder holding many DCH run
  # directories (the normal multi-topic case) or directly at a single run's
  # own output directory (which itself contains a top-level "results/"
  # folder). Without this check, list.dirs() below would enumerate that
  # single run's own results/debug/src folders as if each were a separate
  # hypothesis-pair run, and "results" - which doesn't match the
  # output_<ts>_<topic>_kmgptdch naming topic_from_dirname() expects - would
  # leak into every topic label verbatim (e.g. "results: CHAT").
  if (dir.exists(file.path(proj_path, "results"))) {
    dir_labels <- basename(proj_path)
    dir_paths <- proj_path
  } else {
    dir_labels <- list.dirs(proj_path, full.names = FALSE, recursive = FALSE)
    dir_labels <- dir_labels[!grepl("^output_(model|visualization)_", dir_labels)] # skip our own (and older) output folders
    dir_paths <- file.path(proj_path, dir_labels)
  }

  summaries <- list()
  samples_list <- list()

  for (idx in seq_along(dir_labels)) {
    d <- dir_labels[idx]
    full_dir <- dir_paths[idx]
    groups <- load_hypothesis_groups(full_dir)
    if (is.null(groups)) {
      message(paste0(d, ": no hypothesis-comparison JSON found, skipping"))
      next
    }

    dir_topic <- topic_from_dirname(d)

    # A directory with more than one distinct hypothesis-pair (an A-term-list
    # run) needs a per-pair topic label, not one shared by the whole
    # directory - otherwise every pair's violin would be plotted as if it
    # were the same comparison. Derive that label from what varies across
    # the pairs' hypothesis1 texts (e.g. the A term).
    if (length(groups) > 1) {
      varying_terms <- extract_varying_term(vapply(groups, function(g) g$hypothesis1, character(1)))
    } else {
      varying_terms <- NA_character_
    }

    for (i in seq_along(groups)) {
      g <- groups[[i]]
      labels <- short_hypothesis_labels(g$hypothesis1, g$hypothesis2)
      topic <- if (length(groups) > 1) paste0(dir_topic, ": ", varying_terms[i]) else dir_topic

      result <- summarize_topic(
        topic = topic, hyp1_label = labels$term1, hyp2_label = labels$term2,
        hypothesis1 = g$hypothesis1, hypothesis2 = g$hypothesis2,
        calls = g$calls, level = opt$level, n_samples = opt$n_samples
      )

      message(paste0(
        d, " [", topic, "]: (", labels$term1, " vs ", labels$term2, ") - ",
        length(g$calls), " calls"
      ))

      summaries[[length(summaries) + 1]] <- result$summary
      samples_list[[length(samples_list) + 1]] <- result$samples
    }
  }

  if (length(summaries) == 0) {
    stop("No hypothesis-pair directories with usable data were found under ", proj_path)
  }

  summary_df <- do.call(rbind, summaries)
  samples_df <- do.call(rbind, samples_list)

  write.table(summary_df,
    file = file.path(output_dir, "summary_stats.txt"),
    sep = "\t", col.names = TRUE, quote = FALSE, row.names = FALSE
  )
  write.table(samples_df,
    file = file.path(output_dir, "posterior_samples.txt"),
    sep = "\t", col.names = TRUE, quote = FALSE, row.names = FALSE
  )

  # order topics by posterior mean, low to high (coord_flip below puts the
  # highest mean at the top of the plot)
  summary_df <- summary_df[order(summary_df$posterior_mean), ]
  topic_levels <- summary_df$topic
  summary_df$topic <- factor(summary_df$topic, levels = topic_levels)
  samples_df$topic <- factor(samples_df$topic, levels = topic_levels)

  p <- ggplot(samples_df, aes(x = topic, y = posterior, fill = topic)) +
    geom_violin(trim = FALSE, linewidth = 0.3) +
    geom_pointrange(
      data = summary_df,
      aes(x = topic, y = posterior_mean, ymin = hdi_lo, ymax = hdi_hi, fill = NULL),
      inherit.aes = FALSE, size = 0.3, linewidth = 0.6
    ) +
    geom_hline(yintercept = 50, linetype = "dashed", color = "darkgrey") +
    # H1/H2 labels sit just outside the 0/100 edges (not deep in the margin -
    # just enough to clear a violin whose mean sits close to 0 or 100) -
    # topic name is left as the normal axis label, so it stays on the left
    # where it's always been, and isn't competing with these for space.
    geom_text(
      data = summary_df, aes(x = topic, y = -6, label = hyp2_label),
      inherit.aes = FALSE, size = 2.4, hjust = 1
    ) +
    geom_text(
      data = summary_df, aes(x = topic, y = 106, label = hyp1_label),
      inherit.aes = FALSE, size = 2.4, hjust = 0
    ) +
    scale_fill_viridis(discrete = TRUE, option = "C", guide = "none") +
    scale_y_continuous(breaks = seq(0, 100, 25)) +
    coord_flip(ylim = c(-55, 155), clip = "off") +
    labs(x = NULL, y = paste0("posterior score (mean, ", sprintf("%.0f%%", opt$level * 100), " HDI)")) +
    theme_bw() +
    theme(
      axis.line = element_line(color = "black"),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 15)
    )

  # Side-by-side bar plot of mean supporting-abstract counts per H1/H2,
  # sharing the violin's topic ordering/factor levels so rows line up when
  # the two panels are combined below.
  bar_df <- rbind(
    data.frame(topic = summary_df$topic, term = "H1", count = summary_df$mean_support_h1),
    data.frame(topic = summary_df$topic, term = "H2", count = summary_df$mean_support_h2)
  )
  bar_df$term <- factor(bar_df$term, levels = c("H1", "H2"))

  p_bar <- ggplot(bar_df, aes(x = topic, y = count, fill = term)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c(H1 = "#31688e", H2 = "#8fd744"), name = NULL) +
    coord_flip() +
    labs(x = NULL, y = "mean supporting abstracts") +
    theme_bw() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black"),
      legend.position = "top",
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )

  combined <- p + p_bar + patchwork::plot_layout(widths = c(3, 1))

  plot_height <- max(6, 0.45 * nlevels(summary_df$topic) + 2)
  ggsave(file.path(output_dir, "posterior_violin_plot.pdf"), combined, width = 11, height = plot_height, units = "in")

  cat("Plots and data saved to", output_dir, "\n")
}

if (sys.nframe() == 0) {
  main()
}

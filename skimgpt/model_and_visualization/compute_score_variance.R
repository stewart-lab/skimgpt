#!/usr/bin/env Rscript

# Computes the variance of the Score column for each results.tsv file found
# in the output folders under this directory, then averages the per-file
# variances into one final value.

library(ggplot2)
# base directory is where the run files for the variance test are located
base_dir <- "/w5home/bmoore/Hyp1vsHyp2_paper/new_variance_test/"
#base_dir <- getwd()
results_files <- list.files(
  base_dir,
  pattern = "^results\\.tsv$",
  recursive = TRUE,
  full.names = TRUE
)

if (length(results_files) == 0) {
  stop("No results.tsv files found under ", base_dir)
}

call_variances <- sapply(results_files, function(f) {
  df <- read.delim(f, sep = "\t", header = TRUE, check.names = FALSE)
  var(df$Score / 100, na.rm = TRUE)
})

names(call_variances) <- results_files

cat("Per-file Score variance:\n")
for (f in names(call_variances)) {
  cat(sprintf("  %s: %s\n", basename(dirname(f)), call_variances[[f]]))
}

average_call_variance <- mean(call_variances, na.rm = TRUE)

cat(sprintf("\nAverage variance across %d files: %s\n", length(call_variances), average_call_variance))

# get means
call_means <- sapply(results_files, function(f) {
  df <- read.delim(f, sep = "\t", header = TRUE, check.names = FALSE)
  mean(df$Score / 100, na.rm = TRUE)
})

names(call_means) <- results_files

proportional_var <- sapply(call_means, function(x) {
  x * (1 - x)
})

# confirm all three vectors are keyed by the same set of files before merging
stopifnot(
  setequal(names(call_variances), names(call_means)),
  setequal(names(call_means), names(proportional_var))
)

file_order <- names(call_variances)

# write results
result_df <- data.frame(
  "a_term" = gsub("output_\\d+_(.+?)_kmgptdch_.*", "\\1", basename(dirname(file_order))),
  "score_variance" = as.numeric(call_variances[file_order]),
  "score_mean" = as.numeric(call_means[file_order]),
  "score_proportional_variance" = as.numeric(proportional_var[file_order])
)

print(result_df)
write.csv(result_df, file.path(base_dir, "score_variance_results2.csv"), row.names = FALSE)

# scatter plot of score_variance vs. score_proportional_variance with linear trendline
fit <- lm(score_variance ~ score_proportional_variance, data = result_df)
print(summary(fit))

# use a y-intercept of 0 because variance cannot go below zero
fit0 <- lm(score_variance ~ 0 + score_proportional_variance, data = result_df)
summary(fit0)

eqn_label <- sprintf(
  "y = %.3gx\nAdj. R² = %.3g",
  coef(fit0)[1], summary(fit0)$adj.r.squared
)

p1 <- ggplot(result_df, aes(x = score_proportional_variance, y = score_variance)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", formula = y ~ 0 + x, se = FALSE, color = "red") +
  annotate(
    "text",
    x = -Inf, y = Inf, label = eqn_label,
    hjust = -0.1, vjust = 1.2
  ) +
  labs(
    x = "Proportional variance",
    y = "Score variance",
    title = "Score variance vs. proportional variance"
  ) +
  theme_bw()

plot_file1 <- file.path(base_dir, "score_variance_vs_proportional_variance_y0.pdf")
ggsave(plot_file1, plot = p1, width = 6, height = 5)

cat(sprintf("\nSaved plot1 to %s\n", plot_file1))

# scatter plot of score_variance vs. score_means with linear trendline
fit <- lm(score_variance ~ score_mean, data = result_df)
print(summary(fit))

eqn_label <- sprintf(
  "y = %.3gx + %.3g\nAdj. R² = %.3g",
  coef(fit)[2], coef(fit)[1], summary(fit)$adj.r.squared
)

p2 <- ggplot(result_df, aes(x = score_mean, y = score_variance)) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "blue") +
  annotate(
    "text",
    x = -Inf, y = Inf, label = eqn_label,
    hjust = -0.1, vjust = 1.2
  ) +
  labs(
    x = "Score mean",
    y = "Score variance",
    title = "Score variance vs. Score mean"
  ) +
  theme_bw()

plot_file2 <- file.path(base_dir, "score_variance_vs_score_mean.pdf")
ggsave(plot_file2, plot = p2, width = 6, height = 5)

cat(sprintf("\nSaved plot2 to %s\n", plot_file2))

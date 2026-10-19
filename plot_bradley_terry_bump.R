# Bump / slope charts comparing Bradley-Terry results from the early passes
# (pass 1, or the first N passes) with all passes, and the rank_wrapper
# win-loss ranking with the Bradley-Terry all-passes ranking. Each gene is a
# line from its left position to its right position, coloured by how far it
# moved. If a cell-type file is given, the gene name/rank labels are coloured by
# cell type. Reads the TSVs written by bradleyterry_model.py (and
# FINAL_RANKING.txt) and saves PDFs to the same run directory:
#   bradley_terry_bump_rank.pdf        - y = rank (bump chart)
#   bradley_terry_slope_log2.pdf       - y = log2 strength on a shared axis (slope chart)
#   bradley_terry_vs_winloss_rank.pdf  - win-loss rank vs Bradley-Terry all-passes rank (bump chart)
# The first two get a _first<N>passes suffix when N > 1, and N > 1 also adds
# bradley_terry_vs_winloss_rank_first<N>passes.pdf (win-loss vs first N passes).
#
# Usage: Rscript plot_bradley_terry_bump.R <run_dir> [--out-dir dir] [--initial-passes N] [--celltypes file]

library(ggplot2)
library(ggrepel)

# Shared args, palette and data loading (see plot_bradley_terry_common.R)
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
source(file.path(script_dir, "plot_bradley_terry_common.R"))

# Rank change between the left (first level of `set`) and right result:
# positive = moved up. Genes missing from either side, or with a meaningless
# rank (Rank set to NA in `rank_for_change`), get NA and are drawn in grey.
add_rank_change <- function(df, rank_for_change = df$Rank) {
  sides <- levels(df$set)
  change <- df %>%
    mutate(r = rank_for_change) %>%
    select(Gene, set, r) %>%
    tidyr::pivot_wider(names_from = set, values_from = r) %>%
    transmute(Gene, change = .data[[sides[1]]] - .data[[sides[2]]])
  left_join(df, change, by = "Gene")
}

bt <- bt %>%
  mutate(value = log2(Strength), RankLabel = as.character(Rank)) %>%
  add_rank_change()

plot_bump <- function(bt, y_var, y_lab, title, reverse_y = FALSE) {
  left <- filter(bt, set == levels(bt$set)[1])
  right <- filter(bt, set == levels(bt$set)[2])
  max_change <- max(abs(bt$change), na.rm = TRUE)

  # Labels take a fixed per-gene cell-type colour (not mapped, since the colour
  # scale is used by rank change), otherwise dark grey
  label_layer <- function(data, mapping, nudge, hjust) {
    args <- list(
      data = data, mapping = mapping,
      nudge_x = nudge, direction = "y", hjust = hjust,
      segment.colour = "grey75", segment.size = 0.3, size = 3,
      max.overlaps = Inf, show.legend = FALSE
    )
    args$colour <- if (has_celltypes) unname(celltype_colours[as.character(data$celltype)]) else "grey15"
    do.call(geom_text_repel, args)
  }

  p <- ggplot(bt, aes(x = set, y = .data[[y_var]], group = Gene, colour = change)) +
    geom_line(linewidth = 0.9, alpha = 0.85) +
    geom_point(size = 2.5) +
    label_layer(left, aes(label = paste0(Gene, " (", RankLabel, ")")), nudge = -0.15, hjust = 1) +
    label_layer(right, aes(label = paste0("(", RankLabel, ") ", Gene)), nudge = 0.15, hjust = 0) +
    scale_colour_gradient2(
      low = "#B2182B", mid = "grey70", high = "#2166AC", midpoint = 0,
      limits = c(-max_change, max_change), na.value = "grey40",
      name = "Rank change\n(+ = moved up)"
    ) +
    scale_x_discrete(expand = expansion(add = 0.8)) +
    labs(title = title, x = NULL, y = y_lab) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(face = "bold")
    )

  # Cell-type legend for the label colours, drawn from invisible points
  if (has_celltypes) {
    p <- p +
      geom_point(aes(fill = celltype), shape = 22, size = 0, alpha = 0, show.legend = TRUE) +
      scale_fill_manual(
        values = celltype_colours, drop = TRUE, name = "Label: cell type",
        labels = celltype_labels
      ) +
      guides(fill = guide_legend(override.aes = list(size = 4, alpha = 1, colour = NA)))
  }

  if (reverse_y) p <- p + scale_y_reverse(breaks = c(1, seq(5, max(bt$Rank), by = 5)))
  p
}

p_rank <- plot_bump(
  bt, "Rank", "Rank",
  paste0("Amacrine cell marker ranking: ", early_name, " vs all passes (Bradley-Terry rank)"),
  reverse_y = TRUE
)

p_slope <- plot_bump(
  bt, "value", "log2(Bradley-Terry strength)",
  paste0("Amacrine cell marker ranking: ", early_name, " vs all passes (Bradley-Terry strength)")
) +
  geom_hline(yintercept = 0, colour = "grey60", linetype = "dashed", linewidth = 0.4) +
  labs(caption = paste0(
    "Both results are normalized to their geometric mean, so 0 = average gene in each.\n",
    early_caption, ", so strengths are shrunk toward 0;\n",
    "the wider spread across all passes reflects more evidence, not a change in the genes."
  )) +
  theme(plot.caption = element_text(hjust = 0, colour = "grey40"))

outputs <- list(
  list(file = paste0("bradley_terry_bump_rank", out_suffix, ".pdf"), plot = p_rank),
  list(file = paste0("bradley_terry_slope_log2", out_suffix, ".pdf"), plot = p_slope)
)

# Win-loss ranking (FINAL_RANKING.txt) vs a Bradley-Terry result. Only ranks
# are comparable here. Terms flagged as insufficient evidence / unresolved have
# no meaningful win-loss rank, so they are tied last (e.g. "16=") and their
# lines are grey.
read_winloss <- function(file) {
  lines <- readLines(file)
  entry <- regmatches(lines, regexec("^(\\d+)\\. (.+?) \\(W\\d+-T\\d+-L\\d+", lines, perl = TRUE))
  is_entry <- lengths(entry) == 3
  winloss <- tibble(
    Rank = as.integer(sapply(entry[is_entry], `[`, 2)),
    Gene = sapply(entry[is_entry], `[`, 3),
    not_meaningful = grepl("rank not meaningful", lines[is_entry])
  )
  tied_rank <- suppressWarnings(min(winloss$Rank[winloss$not_meaningful]))
  winloss %>%
    mutate(
      Rank = if_else(not_meaningful, tied_rank, Rank),
      RankLabel = if_else(not_meaningful, paste0(Rank, "="), as.character(Rank)),
      set = "Win-loss record\n(FINAL_RANKING.txt)"
    )
}

plot_winloss <- function(winloss, bt_set, bt_name) {
  bt_side <- bt %>%
    filter(set == bt_set) %>%
    select(Gene, Rank, RankLabel, set) %>%
    mutate(set = as.character(set), not_meaningful = FALSE)

  wl <- bind_rows(winloss, bt_side) %>%
    mutate(set = factor(set, levels = c(winloss$set[1], bt_set)))
  if (has_celltypes) {
    wl <- wl %>%
      left_join(celltypes, by = "Gene") %>%
      mutate(celltype = factor(coalesce(celltype, "unknown"), levels = names(celltype_colours)))
  }
  wl <- add_rank_change(wl, if_else(wl$not_meaningful, NA_real_, as.numeric(wl$Rank)))

  p <- plot_bump(
    wl, "Rank", "Rank",
    paste0("Amacrine cell marker ranking: win-loss record vs Bradley-Terry (", bt_name, ")"),
    reverse_y = TRUE
  )
  if (any(winloss$not_meaningful)) {
    tied_rank <- min(winloss$Rank[winloss$not_meaningful])
    p <- p +
      labs(caption = paste0(
        "Win-loss ranks ", tied_rank, "-", nrow(winloss),
        " are flagged in FINAL_RANKING.txt as not meaningful (e.g. insufficient evidence),\n",
        "so they are tied last (", tied_rank, "=)",
        " and their lines are grey."
      )) +
      theme(plot.caption = element_text(hjust = 0, colour = "grey40"))
  }
  p
}

# Always compare against all passes; with --initial-passes N > 1, also
# against the first-N-passes result
final_ranking_file <- file.path(run_dir, "FINAL_RANKING.txt")
if (file.exists(final_ranking_file)) {
  winloss <- read_winloss(final_ranking_file)
  outputs <- c(outputs, list(list(
    file = "bradley_terry_vs_winloss_rank.pdf",
    plot = plot_winloss(winloss, set_levels[2], "all passes")
  )))
  if (initial_passes > 1) {
    outputs <- c(outputs, list(list(
      file = paste0("bradley_terry_vs_winloss_rank", out_suffix, ".pdf"),
      plot = plot_winloss(winloss, set_levels[1], early_name)
    )))
  }
}

for (out in outputs) {
  out_file <- file.path(out_dir, out$file)
  ggsave(out_file, out$plot, width = 9, height = 10)
  message("Saved to ", out_file)
}

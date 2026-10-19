# Timeline-style plot of Bradley-Terry gene strengths.
# Each gene is a point on a vertical line; the spacing between points
# reflects the difference in strength. Reads the TSVs written by
# bradleyterry_model.py and saves PDFs to the same run directory: one with
# a separate x-axis per result, one with a shared x-axis. If a cell-type file
# is given, the shared-axis plot colours each gene by cell type.
#
# Usage: Rscript plot_bradley_terry_timeline.R <run_dir> [--out-dir dir] [--initial-passes N] [--celltypes file]

library(ggplot2)
library(ggrepel)

# Shared args, palette and data loading (see plot_bradley_terry_common.R)
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))
source(file.path(script_dir, "plot_bradley_terry_common.R"))

# Bradley-Terry strengths are multiplicative (P(i beats j) = p_i / (p_i + p_j)),
# so log2 spacing means equal gaps = equal odds ratios. Set FALSE for raw strength.
use_log <- TRUE

bt <- bt %>%
  mutate(
    value = if (use_log) log2(Strength) else Strength,
    # Alternate labels above/below the line so neighbours don't collide
    side = ifelse(Rank %% 2 == 1, 1, -1)
  )

value_lab <- if (use_log) "log2(Bradley-Terry strength)" else "Bradley-Terry strength"

# Shared axis caption: the wider spread in the all-passes result mostly reflects more
# comparisons outweighing the regularizing tie against the average gene.
centre <- if (use_log) "0" else "1"
shared_caption <- paste0(
  "Both results are normalized to their geometric mean, so ", centre, " = average gene in each.\n",
  early_caption, ", so strengths are shrunk toward ", centre, ";\n",
  "the wider spread across all passes reflects more evidence, not a change in the genes."
)

plot_timeline <- function(bt, shared_axis, by_celltype = FALSE) {
  # Dots (and labels, when colouring by cell type) are coloured either by
  # cell type or by strength.
  colour_var <- if (by_celltype) "celltype" else "value"
  colour_scale <- if (by_celltype) {
    scale_colour_manual(
      values = celltype_colours, drop = TRUE, name = "Cell type",
      labels = celltype_labels
    )
  } else {
    scale_colour_viridis_c(option = "plasma", end = 0.9, guide = "none")
  }
  label_aes <- if (by_celltype) {
    aes(label = paste0(Rank, ". ", Gene), colour = celltype)
  } else {
    aes(label = paste0(Rank, ". ", Gene))
  }

  # Genes sit on a vertical line at x = 0; strength runs up the y-axis.
  # (Aesthetics are swapped rather than using coord_flip(), which doesn't
  # support free facet scales.)
  ggplot(bt, aes(x = 0, y = value)) +
    geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.6) +
    geom_point(aes(colour = .data[[colour_var]]), size = 3) +
    geom_text_repel(
      data = filter(bt, side == 1), mapping = label_aes,
      nudge_x = 0.4, direction = "y", hjust = 0,
      segment.colour = "grey70", segment.size = 0.3, size = 3,
      max.overlaps = Inf, show.legend = FALSE
    ) +
    geom_text_repel(
      data = filter(bt, side == -1), mapping = label_aes,
      nudge_x = -0.4, direction = "y", hjust = 1,
      segment.colour = "grey70", segment.size = 0.3, size = 3,
      max.overlaps = Inf, show.legend = FALSE
    ) +
    colour_scale +
    scale_x_continuous(limits = c(-1.5, 1.5)) +
    facet_wrap(~set, nrow = 1, scales = if (shared_axis) "fixed" else "free_y") +
    labs(
      title = "Amacrine cell marker ranking (Bradley-Terry)",
      x = NULL, y = value_lab,
      caption = if (shared_axis) shared_caption else NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", hjust = 0),
      plot.caption = element_text(hjust = 0, colour = "grey40"),
      legend.position = "top"
    )
}

suffix <- if (use_log) "_log2" else ""
for (shared_axis in c(FALSE, TRUE)) {
  out_file <- file.path(out_dir, paste0(
    "bradley_terry_timeline", suffix, if (shared_axis) "_shared_axis" else "", out_suffix, ".pdf"
  ))
  # Cell-type colouring is applied to the shared-axis plot only
  p <- plot_timeline(bt, shared_axis, by_celltype = shared_axis && has_celltypes)
  ggsave(out_file, p, width = 10, height = 10)
  message("Saved to ", out_file)
}

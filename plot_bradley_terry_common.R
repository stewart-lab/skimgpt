# Shared setup for the Bradley-Terry plotting scripts: command-line arguments,
# the cell-type palette, and loading the TSVs written by bradleyterry_model.py.
#
# Usage: Rscript <plot script> <run_dir> [--out-dir dir] [--initial-passes N] [--celltypes file]
#   --out-dir         folder holding the bradley_terry_*.tsv files, where PDFs are
#                     also saved (default: run_dir)
#   --initial-passes  which early result to compare against all passes
#                     (1 = bradley_terry_initial.tsv, N = bradley_terry_first<N>passes.tsv)
#   --celltypes       tab-separated file with columns gene, celltype
#                     (amacrine / ganglion / both)

library(dplyr)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript <script> <run_dir> [--out-dir dir] [--initial-passes N] [--celltypes file]")
run_dir <- args[1]

get_opt <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
out_dir <- get_opt("--out-dir", run_dir)
initial_passes <- as.integer(get_opt("--initial-passes", "1"))
celltype_file <- get_opt("--celltypes", NA)
has_celltypes <- !is.na(celltype_file)

celltype_colours <- c(
  amacrine = "#2E2A7A",  # dark purple-blue
  ganglion = "#C7860A",  # dark yellow-orange
  both     = "#2E8B3E",  # green
  unknown  = "grey60"
)
celltype_labels <- c(amacrine = "Amacrine", ganglion = "Ganglion", both = "Both", unknown = "Unknown")

# Names for the early result in titles, captions and output files
initial_tsv <- if (initial_passes == 1) "bradley_terry_initial.tsv" else
  paste0("bradley_terry_first", initial_passes, "passes.tsv")
out_suffix <- if (initial_passes == 1) "" else paste0("_first", initial_passes, "passes")
early_name <- if (initial_passes == 1) "initial" else paste("first", initial_passes, "passes")
early_caption <- if (initial_passes == 1) "Pass 1 has one comparison per gene" else
  paste("The first", initial_passes, "passes have few comparisons per gene")

# Both results stacked, with `set` = the result's label (from the TSV's Label
# column, broken before the counts so it fits in facet strips / axis labels)
read_bt <- function(file) {
  read_tsv(file.path(out_dir, file), show_col_types = FALSE) %>%
    mutate(set = sub(" (", "\n(", Label, fixed = TRUE))
}
bt <- bind_rows(read_bt(initial_tsv), read_bt("bradley_terry_all.tsv"))
set_levels <- unique(bt$set)
bt <- mutate(bt, set = factor(set, levels = set_levels))

if (has_celltypes) {
  celltypes <- read_tsv(celltype_file, show_col_types = FALSE) %>%
    transmute(Gene = trimws(gene), celltype = tolower(trimws(celltype)))
  bt <- bt %>%
    left_join(celltypes, by = "Gene") %>%
    mutate(celltype = factor(coalesce(celltype, "unknown"), levels = names(celltype_colours)))
}

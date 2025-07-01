import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib

# ------------------
# Global matplotlib style (match figure1)
# ------------------
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 8,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def read_scores(file_path):
    """Reads scores from a results file."""
    scores = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            header_skipped = False
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                # ---------- Old colon-separated format ----------
                if "Relationship:" in line and "Score:" in line:
                    parts = line.split(":")
                    if len(parts) >= 4:
                        drug_info = parts[1].split(",")[0]
                        drug = drug_info.split("-")[-1].strip()
                        try:
                            score = int(parts[-1].strip())
                            scores[drug] = score
                        except ValueError:
                            print(f"Warning: Could not parse score in line: {line}")
                    continue  # handled

                # ---------- TSV format (tab-separated) ----------
                if not header_skipped and line.lower().startswith("relationship_type"):
                    header_skipped = True
                    continue
                if "\t" in line:
                    t_parts = line.split("\t")
                    if len(t_parts) == 3:
                        _, relationship, score_str = t_parts
                        # drug name is text after last hyphen
                        drug = relationship.split("-")[-1].strip()
                        try:
                            score = int(score_str.strip())
                            scores[drug] = score
                        except ValueError:
                            print(f"Warning: Could not parse integer score in {file_path}: {line}")
                    continue

    except FileNotFoundError:
        print(f"Warning: Results file not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    if not scores:
        print(f"Warning: No scores read from {file_path}")
    return scores

# -------------------------------------------------------------
# Core functionality
# -------------------------------------------------------------

PROMPT_DIRS = [
    "empty_text",
    "negative_text",
    "neutral_text",
    "positive_text",
    "real_text",
]

LABEL_MAP = {
    "negative": "Negative",
    "empty": "No text",
    "neutral": "Neutral",
    "real": "Real text",
    "positive": "Positive",
}

# All categories rendered in the same gray tone
COLOR_PALETTE = {c: "lightgray" for c in [
    "Negative", "No text", "Neutral", "Real text", "Positive"
]}

CATEGORY_ORDER = ["No text", "Negative", "Neutral", "Positive", "Real text"]

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # one level up from paper/

def load_hypothesis_dataframe(base_dir: str) -> pd.DataFrame | None:
    """Load scores for a single hypothesis directory into a melted DataFrame."""
    scores_data: dict[str, dict[str, int]] = {}

    # first try path relative to script directory (paper/<base_dir>)
    abs_base_dir = SCRIPT_DIR / base_dir
    if not abs_base_dir.exists():
        # fallback to project root
        abs_base_dir = PROJECT_ROOT / base_dir

    for directory in PROMPT_DIRS:
        full_dir_path = abs_base_dir / directory
        file_path = full_dir_path / "results.txt"
        if file_path.exists():
            text_type = directory.split("_")[0]
            scores_data[text_type] = read_scores(str(file_path))
        else:
            print(f"Warning: Missing results.txt for {directory} under {abs_base_dir}")

    if not scores_data:
        print(f"Error: No scores found for {base_dir}")
        return None

    df = pd.DataFrame.from_dict(scores_data, orient="index")
    if df.empty:
        print(f"Error: Empty DataFrame for {base_dir}")
        return None

    df.index = df.index.map(lambda x: LABEL_MAP.get(x, x.capitalize()))
    df_melted = df.reset_index().melt(id_vars="index", var_name="Drug", value_name="Score")
    df_melted = df_melted.rename(columns={"index": "Abstract Type"})
    df_melted.dropna(subset=["Score"], inplace=True)
    df_melted["Score"] = df_melted["Score"].astype(int)

    # add small jitter for better strip overlay
    np.random.seed(42)
    df_melted["Jittered_Score"] = pd.to_numeric(df_melted["Score"], errors="coerce") + np.random.uniform(
        -0.1, 0.1, size=len(df_melted)
    )
    df_melted.dropna(subset=["Jittered_Score"], inplace=True)
    return df_melted


def plot_violin(ax: plt.Axes, df: pd.DataFrame, title: str):
    """Render a violin + strip plot on the provided Axes."""
    if df is None or df.empty or "Abstract Type" not in df.columns:
        # No data available: display placeholder text and hide axes
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center", fontsize=8, weight="bold")
        return

    sns.violinplot(
        ax=ax,
        x="Abstract Type",
        y="Jittered_Score",
        data=df,
        palette=COLOR_PALETTE,
        order=CATEGORY_ORDER,
        inner="box",
        cut=0,
        linewidth=1.0,
    )
    sns.stripplot(
        ax=ax,
        x="Abstract Type",
        y="Jittered_Score",
        data=df,
        order=CATEGORY_ORDER,
        color="black",
        alpha=0.6,
        size=3,
        jitter=0.15,
    )
    ax.set_title(title, pad=6)
    ax.set_xlabel("Abstract Sentiment")
    ax.set_ylabel("Support Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)


def parse_latex_table(path: str) -> pd.DataFrame:
    """Parse a simple LaTeX tabular with \hline separated rows into a DataFrame."""
    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("\\hline"):
                continue
            if "&" in line:
                # remove trailing \\\
                line_clean = re.sub(r"\\\\.*", "", line)
                parts = [
                    re.sub(r"\\textbf\{|\}", "", p).strip()
                    for p in line_clean.split("&")
                ]
                rows.append(parts)
    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    header, *data = rows
    return pd.DataFrame(data, columns=header)


def render_table(ax: plt.Axes, df: pd.DataFrame):
    """Render a pandas DataFrame as a matplotlib table in the given Axes."""
    ax.axis("off")
    # Keep only first two columns (Drug & FDA Year)
    df_trim = df.iloc[:, :2]

    # Create table and manually adjust layout so header doesn't overlap
    tbl = ax.table(
        cellText=df_trim.values,
        colLabels=df_trim.columns,
        loc="upper center",
        cellLoc="left",
        bbox=[0, -0.12, 1, 1.12],  # further shift to align with Panel C title
    )

    # Reduce width of the 2nd column (FDA year)
    first_col_w, second_col_w = 0.72, 0.28
    n_rows = df_trim.shape[0] + 1  # +1 for header row (row=0)
    for row in range(n_rows):
        # Column 0
        cell = tbl[(row, 0)]
        cell.set_width(first_col_w)
        # Column 1
        cell = tbl[(row, 1)]
        cell.set_width(second_col_w)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    ax.set_title("D) FDA-approved Breast-Cancer Drugs", pad=6)


def create_combined_figure():
    """Create a 4-panel publication figure (3 hypothesis plots + drug table)."""

    # Mapping of base directories to subplot titles (will add panel letters later)
    hypos = [
        ("leakage_negative_hypo", "Negative hypothesis"),
        ("leakage_neutral_hypo", "Neutral hypothesis"),
        ("leakage_positive_hypo", "Positive hypothesis"),
    ]

    dfs: list[tuple[pd.DataFrame,str]] = []
    panel_letters = ["A", "B", "C"]
    for (base, title), letter in zip(hypos, panel_letters):
        df = load_hypothesis_dataframe(base)
        full_title = f"{letter}) {title}"
        if df is not None:
            dfs.append((df, full_title))
        else:
            dfs.append((pd.DataFrame(), full_title))

    # Ensure table path is resolved relative to script directory
    drug_df = parse_latex_table(str(SCRIPT_DIR / "table2.tex"))

    # ---------------------- 2×2 layout ----------------------
    fig_grid, axes = plt.subplots(2, 2, figsize=(6.69, 8))
    axes = axes.flatten()

    # First three axes: plots
    for idx, (df, title) in enumerate(dfs):
        plot_violin(axes[idx], df, title)

    # Last axis: table
    render_table(axes[3], drug_df)

    fig_grid.tight_layout()
    grid_path = "figure4_combined_2x2_fullpage.pdf"
    fig_grid.savefig(grid_path, dpi=300, bbox_inches="tight", format="pdf", metadata={"Creator": "", "Producer": ""})
    plt.close(fig_grid)

    # ------------------ 4-row single-column ------------------
    fig_col, axes_col = plt.subplots(4, 1, figsize=(3.35, 8), sharex=False)

    for idx, (df, title) in enumerate(dfs):
        plot_violin(axes_col[idx], df, title)

    # Replace last axe with table
    fig_col.delaxes(axes_col[-1])
    ax_table = fig_col.add_subplot(4, 1, 4)
    render_table(ax_table, drug_df)

    fig_col.tight_layout()
    col_path = "figure4_combined_singlecol_halfpage.pdf"
    fig_col.savefig(col_path, dpi=300, bbox_inches="tight", format="pdf", metadata={"Creator": "", "Producer": ""})
    plt.close(fig_col)

    print(f"Saved combined figures: {grid_path} & {col_path}")


if __name__ == "__main__":
    create_combined_figure()
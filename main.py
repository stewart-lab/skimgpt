import itertools
import logging
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

import src.skim_and_km_api as fastkm
from main_wrapper import setup_logger
from src.cost_estimator import KMCostEstimator, SkimCostEstimator
from src.eval_JSON_results import extract_and_write_scores
from src.utils import Config, add_file_handler

logger = logging.getLogger(__name__)


def initialize_workflow() -> tuple[Config, Path]:
    """Set up output directory, copy config, and configure logging."""
    config = Config("config.json")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = Path("output").resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_dir_name = f"output_{timestamp}"
    if config.outdir_suffix:
        timestamp_dir_name = f"{timestamp_dir_name}_{config.outdir_suffix}"
    output_directory = base_output_dir / timestamp_dir_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Copy config and re-initialize from the output directory copy
    config_path = output_directory / "config.json"
    shutil.copy("config.json", config_path)
    config = Config(str(config_path))

    # Set up file-based logging in the output directory
    config.km_output_dir = str(output_directory)
    add_file_handler(str(output_directory))

    logger.info(f"Initializing workflow in {output_directory}")
    return config, output_directory


def main_workflow(combination: dict, output_dir: str, config: Config) -> str | None:
    """Dispatch a single term combination to the appropriate workflow."""
    try:
        workflow_fn = (
            fastkm.skim_with_gpt_workflow
            if config.is_skim_with_gpt
            else fastkm.km_with_gpt_workflow
        )
        return workflow_fn(term=combination, config=config, output_directory=output_dir)
    except Exception as e:
        logger.error(f"Workflow failed for {combination}: {e}")
        return None


def organize_output(directory: Path) -> None:
    """Organize output files into results and debug directories.

    Structure:
        results/ - Final outputs (JSON results, iteration directories)
        debug/   - Intermediate files (TSV files, logs, stderr, stdout)
        config.json - Kept at top level
        results.tsv - Kept at top level
    """
    results_dir = directory / "results"
    debug_dir = directory / "debug"
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    RESULT_SUFFIXES = ("_skim_with_gpt.json", "_km_with_gpt.json", "_km_with_gpt_direct_comp.json")
    DEBUG_EXTENSIONS = (".tsv", ".log", ".err", ".sub", ".out")
    TOP_LEVEL_FILES = {"config.json", "results.tsv"}

    # Move iteration directories into results/
    for item in directory.iterdir():
        if not (item.is_dir() and item.name.startswith("iteration_")):
            continue
        dest_path = results_dir / item.name
        if dest_path.exists():
            # Merge contents into existing destination
            for src_file in item.rglob("*"):
                if not src_file.is_file():
                    continue
                tgt = dest_path / src_file.relative_to(item)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_file), str(tgt))
            shutil.rmtree(str(item), ignore_errors=True)
        else:
            shutil.move(str(item), str(dest_path))
        logger.info(f"Moved {item.name} -> results/{item.name}")

    # Classify remaining files (skip results/ and debug/ subtrees)
    skip_prefixes = (str(results_dir.resolve()), str(debug_dir.resolve()))
    for root, _dirs, files in os.walk(str(directory)):
        if os.path.abspath(root).startswith(skip_prefixes):
            continue
        for f in files:
            file_path = os.path.join(root, f)
            try:
                is_result = any(f.endswith(s) for s in RESULT_SUFFIXES) or f == "no_results.txt"
                is_debug = f.endswith(DEBUG_EXTENSIONS) and f not in TOP_LEVEL_FILES

                if is_result:
                    shutil.move(file_path, str(results_dir / f))
                elif is_debug:
                    shutil.move(file_path, str(debug_dir / f))
                elif f not in TOP_LEVEL_FILES:
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error processing file {f}: {e}")

    # Clean up empty directories (bottom-up)
    for root, dirs, _files in os.walk(str(directory), topdown=False):
        for d in dirs:
            dir_path = Path(root) / d
            try:
                dir_path.rmdir()  # only succeeds if empty
            except OSError:
                pass


def concatenate_tsv_files(tsv_files: list[Path], output_directory: Path) -> tuple[Path, pd.DataFrame, str]:
    """Concatenate TSV files if multiple exist, or return single file."""
    first_filename = tsv_files[0].name
    job_prefix = first_filename.split("_")[0]
    
    if len(tsv_files) > 1:
        logger.info(f"Concatenating {len(tsv_files)} TSV files")
        combined_filename = f"{job_prefix}_combined_output_filtered.tsv"
        combined_tsv_path = output_directory / combined_filename
        
        # Read and concatenate TSV files
        dataframes = [pd.read_csv(tsv, sep="\t") for tsv in tsv_files]
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Write the combined TSV file
        combined_df.to_csv(combined_tsv_path, sep="\t", index=False)
        logger.info(f"Concatenated TSV file saved at {combined_tsv_path}")
    else:
        logger.info("Single TSV file found, no concatenation needed")
        combined_tsv_path = tsv_files[0]
        combined_df = pd.read_csv(combined_tsv_path, sep="\t")
    
    return combined_tsv_path, combined_df, job_prefix


def _get_cost_estimator(config: Config) -> KMCostEstimator | SkimCostEstimator | None:
    """Return the appropriate cost estimator for the current job type."""
    if config.is_km_with_gpt:
        return KMCostEstimator(config)
    if config.is_skim_with_gpt:
        return SkimCostEstimator(config)
    return None


def _get_num_iterations(config: Config) -> int:
    """Return the effective number of iterations (at least 1)."""
    if isinstance(config.iterations, int) and config.iterations > 0:
        return config.iterations
    return 1


def handle_cost_estimation(config: Config, combined_df: pd.DataFrame, output_directory: Path) -> None:
    """Handle cost estimation logic including wrapper handling."""
    logger.info(f"Current job type: {config.job_type}")

    num_iters = _get_num_iterations(config)
    logger.info(f"Calculating cost estimation for {num_iters} iteration(s)...")

    wrapper_parent = os.getenv("WRAPPER_PARENT_DIR")

    if not wrapper_parent:
        # Standalone mode: compute tokens for logging only
        estimator = _get_cost_estimator(config)
        if estimator:
            estimator.estimate_input_costs(combined_df)
        logger.info("Skipping interactive cost prompt (non-interactive mode)")
        return

    sentinel = Path(wrapper_parent) / ".cost_prompt_done"

    if sentinel.is_file():
        # Subsequent wrapper child -- sentinel already exists
        logger.info("Wrapper run detected; skipping per-run cost estimation")
        return

    # First wrapper run: compute total cost across all years, then exit
    yrange = os.getenv("CENSOR_YEAR_RANGE", "")
    yinc = int(os.getenv("CENSOR_YEAR_INCREMENT", "1"))
    lo, hi = map(int, yrange.split("-"))
    num_years = len(range(lo, hi + 1, yinc))

    estimator = _get_cost_estimator(config)

    # Write sentinel so wrapper can launch parallel children
    sentinel.write_text("ok\n")

    setup_logger(wrapper_parent, config.job_type)

    if estimator:
        input_tokens = estimator.estimate_input_costs(combined_df)
        output_tokens = estimator.get_output_tokens()
        in_cost = estimator._calculate_cost(input_tokens, is_input=True)
        out_cost = estimator._calculate_cost(output_tokens, is_input=False)
        cost_per_iter = in_cost + out_cost
        total_cost = cost_per_iter * num_iters * num_years

        logger.info(
            f"Wrapper total estimated cost: ${total_cost:.2f} "
            f"({num_years} years x {num_iters} iters at ${cost_per_iter:.2f}/iteration)"
        )
    else:
        logger.warning(f"Skipping cost estimation for job type: {config.job_type}")

    # Exit so wrapper can kick off the real parallel runs
    sys.exit(0)


def run_relevance_filtering(combined_tsv_path: Path, config: Config) -> None:
    """Run relevance filtering via Triton (with automatic CHTC fallback)."""
    from src.relevance_triton import run_relevance_analysis

    logger.info("Running relevance filtering...")
    try:
        run_relevance_analysis(config, str(combined_tsv_path))
        logger.info("Relevance filtering completed successfully")
    except Exception as e:
        logger.error(f"Error running relevance filtering: {e}", exc_info=True)
        raise


def finalize_results(output_directory: Path, fastkm_start_time: float, rel_and_api_start_time: float) -> None:
    """Organize output, extract scores, and log timing information."""
    logger.info("Processing results...")
    organize_output(output_directory)
    extract_and_write_scores(str(output_directory))

    logger.info(f"Analysis complete. Results are in {output_directory}")
    elapsed_rel_and_api = time.time() - rel_and_api_start_time
    logger.info(f"Relevance and API complete in {elapsed_rel_and_api:.2f} seconds.")
    total_time = time.time() - fastkm_start_time
    logger.info(f"Total time taken: {total_time:.2f} seconds")


def _build_terms(config: Config) -> list[dict] | None:
    """Build the list of term combinations based on job type and position mode.

    Returns None if validation fails (error already logged).
    """
    a_terms = config.a_terms
    b_terms = config.b_terms

    if config.is_skim_with_gpt:
        c_terms = config.c_terms
        if config.position:
            if not (len(a_terms) == len(b_terms) == len(c_terms)):
                logger.error("A, B, and C terms must be the same length for positional mapping.")
                return None
            return [
                {"a_term": a, "b_terms": [b], "c_term": c}
                for a, b, c in zip(a_terms, b_terms, c_terms)
            ]
        # Cartesian product: A x C with all B terms
        return [
            {"a_term": a, "b_terms": b_terms, "c_term": c}
            for a, c in itertools.product(a_terms, c_terms)
        ]

    # km_with_gpt
    if config.position:
        if len(a_terms) != len(b_terms):
            logger.error("A and B terms must be the same length for positional mapping.")
            return None
        return [
            {"a_term": a, "b_terms": [b]}
            for a, b in zip(a_terms, b_terms)
        ]
    return [
        {"a_term": a, "b_terms": b_terms}
        for a in a_terms
    ]


def _flatten_file_paths(raw_paths: list) -> list[Path]:
    """Flatten workflow results into a list of resolved Path objects."""
    result = []
    for item in raw_paths:
        if not item:
            continue
        if isinstance(item, list):
            result.extend(Path(p).resolve() for p in item)
        else:
            result.append(Path(item).resolve())
    return result


def main() -> None:
    fastkm_start_time = time.time()

    config, output_directory = initialize_workflow()
    logger.info("Loading term lists...")
    config.load_term_lists()

    terms = _build_terms(config)
    if terms is None:
        return

    workflow = partial(main_workflow, output_dir=str(output_directory), config=config)

    with multiprocessing.Pool() as p:
        generated_file_paths = p.map(workflow, terms)

    elapsed_fastkm = time.time() - fastkm_start_time
    logger.info(f"fastkm results returned in {elapsed_fastkm:.2f} seconds.")
    rel_and_api_start_time = time.time()

    try:
        output_directory = output_directory.resolve()

        flattened_file_paths = _flatten_file_paths(generated_file_paths)
        if not flattened_file_paths:
            logger.error("No files to process")
            raise ValueError("No files to process")

        logger.info(f"Processing {len(flattened_file_paths)} TSV file(s)")

        combined_tsv_path, combined_df, job_prefix = concatenate_tsv_files(
            flattened_file_paths, output_directory
        )

        # May exit the process if this is the first wrapper run
        handle_cost_estimation(config, combined_df, output_directory)

        run_relevance_filtering(combined_tsv_path, config)
        finalize_results(output_directory, fastkm_start_time, rel_and_api_start_time)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

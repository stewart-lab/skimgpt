from datetime import datetime
from functools import partial
from glob import glob
from src.eval_JSON_results import extract_and_write_scores
from src.jobs import main_workflow
from src.utils import Config
from src.cost_estimator import (
    calculate_total_cost_and_prompt,
    KMCostEstimator,
    SkimCostEstimator,
    WrapperCostEstimator
)
import itertools
import multiprocessing
import os
import pandas as pd
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from main_wrapper import setup_logger




def initialize_workflow():
    global logger
    # Initialize config once so we can get the outdir_suffix
    config = Config("config.json")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = Path("output").resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_dir_name = f"output_{timestamp}"
    if config.outdir_suffix != "":
        timestamp_dir_name = f"{timestamp_dir_name}_{config.outdir_suffix}" 
    output_directory = base_output_dir / timestamp_dir_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    config_path = output_directory / "config.json"
    shutil.copy("config.json", config_path)
    
    # Initialize config again, with the new output directory
    config = Config(str(config_path))
    
    # Set km_output_dir to enable log file creation
    config.km_output_dir = str(output_directory)
    
    # Call add_file_handler to set up logging to a file in the output directory
    config.add_file_handler()
    
    logger = config.logger

    logger.info(f"Initializing workflow in {output_directory}")
    return config, output_directory, logger



def organize_output(directory: Path):
    """Organize output files into results and debug directories.
    
    Structure:
        results/ - Final outputs (JSON results, iteration directories)
        debug/   - Intermediate files (TSV files, logs)
        config.json - Kept at top level
    """
    results_dir = directory / "results"
    debug_dir = directory / "debug"
    results_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Move iteration directories (only from top level)
    for item in directory.iterdir():
        if item.is_dir() and item.name.startswith("iteration_"):
            dest_path = results_dir / item.name
            logger.info(f"Moving iteration directory {item} to {dest_path}")
            shutil.move(str(item), str(dest_path))
    
    # Move loose files in the top level directory
    result_patterns = ["_skim_with_gpt.json", "_km_with_gpt.json", "_km_with_gpt_direct_comp.json"]
    
    for item in directory.iterdir():
        if not item.is_file():
            continue
            
        try:
            # Check if it's a result JSON file
            is_result_json = any(item.name.endswith(pattern) for pattern in result_patterns)
            
            if is_result_json:
                shutil.move(str(item), str(results_dir / item.name))
                logger.info(f"Moved result JSON {item.name} to results/")
            elif item.name == "no_results.txt":
                shutil.move(str(item), str(results_dir / item.name))
            elif item.suffix in (".tsv", ".log"):
                # Only TSV and log files (removed HTCondor-specific: .err, .sub, .out)
                shutil.move(str(item), str(debug_dir / item.name))
                logger.info(f"Moved {item.name} to debug/")
            elif item.name != "config.json":
                # Remove any other files that aren't config.json
                item.unlink()
        except Exception as e:
            logger.error(f"Error processing file {item.name}: {str(e)}")
            continue


def concatenate_tsv_files(tsv_files: List[Path], output_directory: Path, logger) -> Tuple[Path, pd.DataFrame, str]:
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


def handle_cost_estimation(config, combined_df: pd.DataFrame, output_directory: Path, logger):
    """Handle cost estimation logic including wrapper handling."""
    logger.info(f"Current job type: {config.job_type}")
    logger.info("Attempting cost estimation...")

    # Iteration info for cost estimation
    iterations_info = ""
    if config.iterations:
        if isinstance(config.iterations, int) and config.iterations > 0:
            iterations_info = f" for {config.iterations} iterations"
        elif isinstance(config.iterations, bool) and config.iterations:
            iterations_info = " (iterations=True, assuming 1 iteration)"
    logger.info(f"Calculating cost estimation{iterations_info}...")

    wrapper_parent = os.getenv("WRAPPER_PARENT_DIR")
    sentinel = Path(wrapper_parent or "") / ".cost_prompt_done" if wrapper_parent else None

    if wrapper_parent and sentinel and not sentinel.is_file():
        # --- FIRST wrapper-run only: compute tokens & prompt total cost ---
        # figure out how many years in the wrapper
        yrange = os.getenv("CENSOR_YEAR_RANGE", "")
        yinc = int(os.getenv("CENSOR_YEAR_INCREMENT", "1"))
        lo, hi = map(int, yrange.split("-"))
        num_years = len(range(lo, hi + 1, yinc))

        # compute tokens once
        if config.is_km_with_gpt:
            input_tokens = KMCostEstimator(config).estimate_input_costs(combined_df)
            base_est = KMCostEstimator(config)
        elif config.is_skim_with_gpt:
            input_tokens = SkimCostEstimator(config).estimate_input_costs(combined_df)
            base_est = SkimCostEstimator(config)
        else:
            input_tokens = 0
            base_est = None

        # Non-interactive: compute estimate and proceed without prompting
        wrapper_est = WrapperCostEstimator(config)

        # Create sentinel to indicate consent and allow wrapper to launch children
        try:
            with open(sentinel, "w") as _f:
                _f.write("ok\n")
        except Exception:
            pass

        # Re-instantiate your wrapper logger
        wrapper_logger = setup_logger(wrapper_parent, config.job_type)

        # Recompute what you showed the user
        output_tokens = base_est.get_output_tokens()
        in_cost = base_est._calculate_cost(input_tokens, is_input=True)
        out_cost = base_est._calculate_cost(output_tokens, is_input=False)
        cost_per_iter = in_cost + out_cost
        num_iters = (
            config.iterations
            if isinstance(config.iterations, int) and config.iterations > 0
            else 1
        )
        total_cost = cost_per_iter * num_iters * num_years

        wrapper_logger.info(
            f"Wrapper total estimated cost: ${total_cost:.2f} "
            f"({num_years} years x {num_iters} iters at ${cost_per_iter:.2f}/iteration)"
        )

        # Now exit so wrapper can kick off the real parallel runs
        sys.exit(0)

    elif wrapper_parent:
        # any subsequent wrapper child sees sentinel → skip any cost logic
        logger.info("Wrapper run detected; skipping per-run cost estimation")

    else:
        # Non-interactive mode: compute tokens (for logs) and continue without prompt
        if config.is_km_with_gpt:
            _ = KMCostEstimator(config).estimate_input_costs(combined_df)
        elif config.is_skim_with_gpt:
            _ = SkimCostEstimator(config).estimate_input_costs(combined_df)
        logger.info("Skipping interactive cost prompt (non-interactive mode)")


def run_relevance_filtering(combined_tsv_path: Path, config, logger):
    """Run relevance filtering directly."""
    from src.relevance import run_relevance_analysis
    
    logger.info("Running relevance filtering...")
    try:
        run_relevance_analysis(config, str(combined_tsv_path))
        logger.info("Relevance filtering completed successfully")
    except Exception as e:
        logger.error(f"Error running relevance filtering: {str(e)}", exc_info=True)
        raise


def finalize_results(output_directory: Path, logger, fastkm_start_time: float, rel_and_api_start_time: float):
    """Organize output, extract scores, and log timing information."""
    logger.info("Processing results...")
    organize_output(output_directory)
    extract_and_write_scores(str(output_directory))
    
    logger.info(f"Analysis complete. Results are in {output_directory}")
    rel_and_api_end_time = time.time()
    elapsed_rel_and_api_time = rel_and_api_end_time - rel_and_api_start_time
    logger.info(f"Relevance and API complete in {elapsed_rel_and_api_time:.2f} seconds.")
    total_time = time.time() - fastkm_start_time
    logger.info(f"Total time taken: {total_time:.2f} seconds")



def main():
    fastkm_start_time = time.time()

    # Initialize workflow and get config
    config, output_directory, logger = initialize_workflow()
    config.logger.info("Loading term lists...")
    config._load_term_lists()

    # Unified term handling for both job types
    if config.is_skim_with_gpt:
        # Use original term paths without output directory modification
        a_terms = config.a_terms
        b_terms = config.b_terms
        c_terms = config.c_terms
        
        terms = []
        if config.position:
            # make sure the terms are the same length
            if len(a_terms) != len(b_terms) or len(a_terms) != len(c_terms):
                logger.error("A, B, and C terms must be the same length for positional mapping.")
                return
            # Positional mapping: A1-B1-C1, A2-B2-C2
            for a, b, c in zip(a_terms, b_terms, c_terms):
                terms.append({
                    "a_term": a,
                    "b_terms": [b],  # Single B term
                    "c_term": c
                })
        else:
            # Cartesian product: A×C with all B terms
            for a, c in itertools.product(a_terms, c_terms):
                terms.append({
                    "a_term": a,
                    "b_terms": b_terms,  # All B terms
                    "c_term": c
                })
    elif config.is_km_with_gpt:
        # KM workflow - Fix: Use config.a_terms and config.b_terms
        a_terms = config.a_terms
        b_terms = config.b_terms
        
        if config.position:
            # make sure the terms are the same length
            if len(a_terms) != len(b_terms):
                logger.error("A and B terms must be the same length for positional mapping.")
                return
            # Pair A terms with B terms by index (A1-B1, A2-B2)
            terms = [
                {
                    "a_term": a_term,
                    "b_terms": [b_term]  # Single B term per A term
                }
                for a_term, b_term in zip(a_terms, b_terms)
            ]
        else:
            # Original behavior: All B terms for each A term
            terms = [
                {
                    "a_term": a_term,
                    "b_terms": b_terms
                }
                for a_term in a_terms
            ]
    # Removed legacy km_with_gpt_direct_comp branch

    # Maintain the parallel execution pattern
    workflow = partial(
        main_workflow,
        output_dir=str(output_directory),
        config=config
    )

    with multiprocessing.Pool() as p:
        generated_file_paths = p.map(workflow, terms)

    fastkm_end_time = time.time()
    elapsed_fastkm_time = fastkm_end_time - fastkm_start_time

    logger.info(f"fastkm results returned in {elapsed_fastkm_time:.2f} seconds.")
    rel_and_api_start_time = time.time()
    
    try:
        # Ensure we're working with absolute paths
        output_directory = output_directory.resolve()
        
        # Flatten and resolve file paths
        flattened_file_paths = []
        for item in generated_file_paths:
            if item:
                if isinstance(item, list):
                    flattened_file_paths.extend([Path(p).resolve() for p in item])
                else:
                    flattened_file_paths.append(Path(item).resolve())
        
        # Validate we have files to process
        if not flattened_file_paths:
            logger.error("No files to process")
            raise ValueError("No files to process")
        
        logger.info(f"Processing {len(flattened_file_paths)} TSV file(s)")

        # Concatenate TSV files if needed
        combined_tsv_path, combined_df, job_prefix = concatenate_tsv_files(
            flattened_file_paths, output_directory, logger
        )
        
        # Handle cost estimation (may exit if wrapper parent)
        handle_cost_estimation(config, combined_df, output_directory, logger)
        
        # Run relevance filtering
        run_relevance_filtering(combined_tsv_path, config, logger)

        # Finalize results
        finalize_results(output_directory, logger, fastkm_start_time, rel_and_api_start_time)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

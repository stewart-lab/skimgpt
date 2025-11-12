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
import subprocess
from pathlib import Path
from main_wrapper import setup_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def initialize_workflow():
    global logger
    # Initialize config once so we can get the outdir_suffix
    config = Config("config.json")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = os.path.abspath("output")
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"output_{timestamp}"
    if config.outdir_suffix != "":
        timestamp_dir_name = f"{timestamp_dir_name}_{config.outdir_suffix}" 
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)

    # Copy config to output directory
    config_path = os.path.join(output_directory, "config.json")
    shutil.copy("config.json", config_path)
    
    # Initialize config again, with the new output directory
    config = Config(config_path)
    
    # Set km_output_dir to enable log file creation
    config.km_output_dir = output_directory
    
    # Call add_file_handler to set up logging to a file in the output directory
    config.add_file_handler()
    
    logger = config.logger

    logger.info(f"Initializing workflow in {output_directory}")
    return config, output_directory, logger



def organize_output(directory):
    results_dir = os.path.join(directory, "results")
    debug_dir = os.path.join(directory, "debug")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    processed_dirs = set()

    for root, dirs, _ in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith("iteration_"):
                src_path = os.path.join(root, dir_name)
                # Skip if we've already processed this directory
                if src_path in processed_dirs:
                    continue
                
                processed_dirs.add(src_path)
                logger.info(f"Found iteration directory: {src_path}")
                
                # Create the target path in results/
                dest_path = os.path.join(results_dir, dir_name)
                
                if os.path.exists(dest_path):
                    # If destination exists, move content file by file
                    logger.info(f"Destination {dest_path} exists, merging content")
                    for src_root, _, files_dir in os.walk(src_path):
                        rel_root = os.path.relpath(src_root, src_path)
                        target_root = os.path.join(dest_path, rel_root)
                        os.makedirs(target_root, exist_ok=True)
                        for f in files_dir:
                            src_file = os.path.join(src_root, f)
                            dst_file = os.path.join(target_root, f)
                            logger.debug(f"Moving {src_file} to {dst_file}")
                            shutil.move(src_file, dst_file)
                    # After moving files, remove the source directory
                    shutil.rmtree(src_path, ignore_errors=True)
                else:
                    # If destination doesn't exist, move the whole directory
                    logger.info(f"Moving iteration directory {src_path} to {dest_path}")
                    shutil.move(src_path, dest_path)
    
    # Define patterns for result JSON files
    result_patterns = ["_skim_with_gpt.json", "_km_with_gpt.json", "_km_with_gpt_direct_comp.json"]

    for root, dirs, files in os.walk(directory):
        # Skip traversal into results or debug to avoid duplicate moves
        if os.path.abspath(root).startswith(os.path.abspath(results_dir)) or os.path.abspath(root).startswith(os.path.abspath(debug_dir)):
            continue
        
        for file in files:
            try:
                file_path = os.path.join(root, file)
                # Check if it's a result JSON file
                is_result_json = any(file.endswith(pattern) for pattern in result_patterns)
                
                if is_result_json:
                    shutil.move(file_path, os.path.join(results_dir, file))
                elif file == "no_results.txt":
                    shutil.move(file_path, os.path.join(results_dir, file))
                elif file.endswith((".tsv", ".log", ".err", ".sub", ".out")):
                    shutil.move(file_path, os.path.join(debug_dir, file))
                elif file != "config.json":
                    # Remove any other files that aren't config.json
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                continue

    # Clean up empty directories
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            try:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                logger.error(f"Error removing directory {dir}: {str(e)}")
                continue

    filtered_dir = os.path.join(directory, "filtered")
    if os.path.exists(filtered_dir):
        try:
            shutil.rmtree(filtered_dir)
        except Exception as e:
            logger.error(f"Error removing filtered directory: {str(e)}")





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
        # KM workflow
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
                for a_term, b_term in zip(config.a_terms, config.b_terms)
            ]
        else:
            # Original behavior: All B terms for each A term
            terms = [
                {
                    "a_term": a_term,
                    "b_terms": config.b_terms
                }
                for a_term in config.a_terms
            ]
    # Removed legacy km_with_gpt_direct_comp branch

    # Maintain the parallel execution pattern
    workflow = partial(
        main_workflow,
        output_dir=output_directory,
        config=config
    )

    with multiprocessing.Pool() as p:
        generated_file_paths = p.map(workflow, terms)

    
    fastkm_end_time = time.time()
    elapsed_fastkm_time = fastkm_end_time - fastkm_start_time

    logger.info(f"fastkm results returned in {elapsed_fastkm_time:.2f} seconds.")
    rel_and_api_start_time = time.time()
    
    try:
        # Create output directories
        output_results_dir = os.path.join(output_directory, "output")
        os.makedirs(output_results_dir, exist_ok=True)
        
        # Ensure we're working with absolute paths
        output_directory = os.path.abspath(output_directory)
        
        # Flatten and resolve file paths
        flattened_file_paths = []
        for item in generated_file_paths:
            if item:
                if isinstance(item, list):
                    flattened_file_paths.extend([os.path.abspath(p) for p in item])
                else:
                    flattened_file_paths.append(os.path.abspath(item))
        
        if not flattened_file_paths:
            logger.error("No files to process")
            raise ValueError("No files to process")

        # Create files.txt for HTCondor queue
        files_txt_path = os.path.join(output_directory, "files.txt")
        logger.info(f"Creating files.txt at {files_txt_path}")

        # Write to files.txt first
        with open(files_txt_path, "w") as f:
            for file_path in flattened_file_paths:
                filename = os.path.basename(file_path)
                f.write(f"{filename}\n")
                logger.debug(f"Added {filename} to files.txt")

        # Verify files.txt was created
        if not os.path.exists(files_txt_path):
            raise FileNotFoundError(f"Failed to create {files_txt_path}")
        
        with open(files_txt_path, "r") as f:
            tsv_files = [os.path.join(output_directory, line.strip()) for line in f if line.strip()]

        try:
            first_filename = os.path.basename(tsv_files[0])
            job_prefix = first_filename.split("_")[0]
            
            if len(tsv_files) > 1:
                logger.info(f"Concatenating {len(tsv_files)} TSV files")
                combined_filename = f"{job_prefix}_combined_output_filtered.tsv"
                combined_tsv_path = os.path.join(output_directory, combined_filename)
                
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

            # Update files.txt to contain only the combined/single file
            with open(files_txt_path, "w") as f:
                f.write(os.path.basename(combined_tsv_path) + "\n")
                
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
            sentinel      = os.path.join(wrapper_parent or "", ".cost_prompt_done")

            if wrapper_parent and not os.path.isfile(sentinel):
                # --- FIRST wrapper-run only: compute tokens & prompt total cost ---
                # figure out how many years in the wrapper
                yrange = os.getenv("CENSOR_YEAR_RANGE", "")
                yinc   = int(os.getenv("CENSOR_YEAR_INCREMENT", "1"))
                lo, hi = map(int, yrange.split("-"))
                num_years = len(range(lo, hi + 1, yinc))

                # compute tokens once
                if config.is_km_with_gpt:
                    input_tokens = KMCostEstimator(config).estimate_input_costs(combined_df)
                    base_est     = KMCostEstimator(config)
                elif config.is_skim_with_gpt:
                    input_tokens = SkimCostEstimator(config).estimate_input_costs(combined_df)
                    base_est     = SkimCostEstimator(config)
                else:
                    input_tokens = 0
                    base_est     = None

                # Non-interactive: compute estimate and proceed without prompting
                wrapper_est = WrapperCostEstimator(config)

                # Create sentinel to indicate consent and allow wrapper to launch children
                try:
                    with open(sentinel, "w") as _f:
                        _f.write("ok\n")
                except Exception:
                    pass

                # Re-instantiate your wrapper logger
                wrapper_parent = os.getenv("WRAPPER_PARENT_DIR")
                wrapper_logger = setup_logger(wrapper_parent, config.job_type)

                # Recompute what you showed the user
                output_tokens = base_est.get_output_tokens()    # you already have base_est
                in_cost       = base_est._calculate_cost(input_tokens, is_input=True)
                out_cost      = base_est._calculate_cost(output_tokens, is_input=False)
                cost_per_iter = in_cost + out_cost
                num_iters     = (
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
            
        except Exception as e:
            logger.error(f"Error processing TSV files: {str(e)}", exc_info=True)

        # Run relevance filtering directly on the combined TSV file
        logger.info("Running relevance filtering...")
        try:
            config_path = os.path.join(output_directory, "config.json")
            secrets_path = config.secrets_file
            
            # Run relevance.py as a subprocess
            relevance_cmd = [
                sys.executable,
                "-m", "src.relevance",
                "--km_output", combined_tsv_path,
                "--config", config_path,
                "--secrets", secrets_path
            ]
            
            logger.info(f"Running command: {' '.join(relevance_cmd)}")
            result = subprocess.run(
                relevance_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            logger.info("Relevance filtering completed successfully")
            if result.stdout:
                logger.debug(f"Relevance output: {result.stdout}")
            if result.stderr:
                logger.debug(f"Relevance stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Relevance filtering failed with exit code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error running relevance filtering: {str(e)}", exc_info=True)
            raise

        # Process results
        logger.info("Processing results...")
        organize_output(output_directory)
        extract_and_write_scores(output_directory)
        
        logger.info(f"Analysis complete. Results are in {output_directory}")
        rel_and_api_end_time = time.time()
        elapsed_rel_and_api_time = rel_and_api_end_time - rel_and_api_start_time
        logger.info(f"Relevance and API complete in {elapsed_rel_and_api_time:.2f} seconds.")
        total_time = time.time() - fastkm_start_time
        logger.info(f"Total time taken: {total_time:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

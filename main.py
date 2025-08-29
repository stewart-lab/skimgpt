from datetime import datetime
from functools import partial
from glob import glob
from src.eval_JSON_results import extract_and_write_scores
from src.jobs import main_workflow
from src.utils import Config
from src.htcondor_helper import HTCondorHelper
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
from main_wrapper import setup_logger
import tempfile  # used for per-job token directory

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


def write_token_to_file():
    """Create a *new* per-job token directory (no backward compatibility path checks)."""

    token = os.getenv('HTCONDOR_TOKEN')
    if not token:
        raise ValueError("HTCONDOR_TOKEN environment variable not set")

    # Always create under ./token_dirs relative to current working directory
    root_dir = os.path.join(os.getcwd(), 'token_dirs')
    os.makedirs(root_dir, exist_ok=True)

    # Create an exclusive per-job directory (timestamp + pid)
    ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
    token_dir = os.path.join(root_dir, f'token_{ts}_{os.getpid()}')
    os.makedirs(token_dir, exist_ok=True)

    # Point HTCondor to this directory
    os.environ['HTCONDOR_TOKEN_DIR'] = token_dir

    token_file = os.path.join(token_dir, 'condor_token')
    with open(token_file, 'w') as f:
        f.write(token)

    # Restrict permissions
    os.chmod(token_file, 0o600)

    return token_file


def remove_token_file(token_file):
    """Recursively remove the job-specific token directory for security."""
    token_dir = os.path.dirname(token_file)
    if os.path.exists(token_dir):
        shutil.rmtree(token_dir, ignore_errors=True)



def main():
    fastkm_start_time = time.time()
    

    # Initialize workflow and get config
    config, output_directory, logger = initialize_workflow()
    config.logger.info("Loading term lists...")
    config._load_term_lists()

    # Unified term handling for both job types
    if config.job_type == "skim_with_gpt":
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
    elif config.job_type == "km_with_gpt":
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
    if not config.using_htcondor:
        logger.error("HTCONDOR configuration is required but not found in config file.")
        return

    token_file = None
    try:
        # Write token to file
        token_file = write_token_to_file()
        token_dir = os.path.dirname(token_file)
        logger.info("HTCondor token written to token directory")
        
        # Initialize HTCondor helper (connection is tested during initialization)
        try:
            htcondor_helper = HTCondorHelper(config, token_dir)
        except Exception as e:
            logger.error(f"Failed to initialize HTCondor helper: {e}")
            logger.error("Check token and network access.")
            remove_token_file(token_file)
            return

        # Create src directory in output directory
        output_src_dir = os.path.join(output_directory, "src")  
        output_results_dir = os.path.join(output_directory, "output")
        os.makedirs(output_src_dir, exist_ok=True)
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
                if config.job_type in ["km_with_gpt","km_with_gpt_direct_comp"]:
                    input_tokens = KMCostEstimator(config).estimate_input_costs(combined_df)
                    base_est     = KMCostEstimator(config)
                elif config.job_type == "skim_with_gpt":
                    input_tokens = SkimCostEstimator(config).estimate_input_costs(combined_df)
                    base_est     = SkimCostEstimator(config)
                else:
                    input_tokens = 0
                    base_est     = None

                wrapper_est = WrapperCostEstimator(config)
                if not wrapper_est.prompt_total_cost(input_tokens, num_years):
                    logger.info("User aborted wrapper cost prompt")
                    sys.exit(1)

                # user said 'yes' → sentinel is written

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
                # compute tokens and prompt as before
                if config.job_type in ["km_with_gpt","km_with_gpt_direct_comp"]:
                    input_tokens = KMCostEstimator(config).estimate_input_costs(combined_df)
                elif config.job_type == "skim_with_gpt":
                    input_tokens = SkimCostEstimator(config).estimate_input_costs(combined_df)
                else:
                    input_tokens = 0

                if not calculate_total_cost_and_prompt(config, input_tokens):
                    logger.info("Job aborted by user")
                    sys.exit(1)
            
        except Exception as e:
            logger.error(f"Error processing TSV files: {str(e)}", exc_info=True)

        # Copy necessary files to output directory
        src_dir = os.path.join(os.getcwd(), "src")
        for file in ["run.sh", "relevance.py"]:
            src_path = os.path.abspath(os.path.join(src_dir, file))
            dst_path = os.path.join(output_directory, file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logger.debug(f"Copied {file} to {dst_path}")
            else:
                logger.error(f"Required file {file} not found in src directory")
                raise FileNotFoundError(f"Required file {file} not found in {src_dir}")
                
        # Copy source files to src directory
        for src_file in glob(os.path.join(src_dir, "*.py")):
            dst_path = os.path.join(output_src_dir, os.path.basename(src_file))
            if os.path.abspath(src_file) != os.path.abspath(dst_path):
                shutil.copy2(src_file, dst_path)
        
        # Copy input files to output directory
        for src_path in flattened_file_paths:
            dst_path = os.path.abspath(os.path.join(output_directory, os.path.basename(src_path)))
            if src_path != dst_path and os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except shutil.SameFileError:
                    logger.debug(f"Skipping copy of {src_path} as it's already in the destination")
                    continue
        
        # Submit jobs from the output directory
        original_dir = os.getcwd()
        os.chdir(output_directory)
        try:
            cluster_id = htcondor_helper.submit_jobs(files_txt_path)
            monitoring_success = False
            try:
                monitoring_success = htcondor_helper.monitor_jobs(cluster_id)
                if monitoring_success:
                    logger.info("All jobs completed successfully")
                else:
                    logger.warning("Job monitoring ended with some jobs potentially incomplete")
            except Exception as monitor_err:
                logger.error(f"Error during job monitoring: {str(monitor_err)}")
                # Try to release any held jobs if monitoring failed
                try:
                    htcondor_helper.release_held_jobs(cluster_id)
                except Exception:
                    pass

            # Retrieve output regardless of monitoring success
            try:
                logger.info("Retrieving output files...")
                htcondor_helper.retrieve_output(cluster_id)
                logger.info("Output files retrieved successfully")
            except Exception as retrieve_err:
                logger.error(f"Error retrieving output: {str(retrieve_err)}")

            # Process results
            logger.info("Processing results...")
            # recursively dump and print the output directory
            logger.info(f"Recursively dumping output directory: {output_directory}")

            organize_output(output_directory)
            extract_and_write_scores(output_directory)
            
            # Proper cleanup with comprehensive error handling
            try:
                logger.info(f"Cleaning up cluster {cluster_id}...")
                htcondor_helper.cleanup(cluster_id)
                logger.info("Cleanup completed")
            except Exception as cleanup_err:
                logger.error(f"Error during cleanup: {str(cleanup_err)}")
                
        finally:
            os.chdir(original_dir)
            
            # Clean up token file for security
            if token_file:
                remove_token_file(token_file)
                logger.info("HTCondor token file removed for security")
        
        logger.info(f"Analysis complete. Results are in {output_directory}")
        rel_and_api_end_time = time.time()
        elapsed_rel_and_api_time = rel_and_api_end_time - rel_and_api_start_time
        logger.info(f"Relevance and API complete in {elapsed_rel_and_api_time:.2f} seconds.")
        total_time = time.time() - fastkm_start_time
        logger.info(f"Total time taken: {total_time:.2f} seconds")
    except Exception as e:
        # Ensure token is always removed even if an error occurs
        if token_file:
            remove_token_file(token_file)
            logger.info("HTCondor token file removed for security")
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

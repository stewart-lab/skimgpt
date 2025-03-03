from datetime import datetime
from functools import partial
from src.eval_JSON_results import extract_and_write_scores
import shutil
import itertools
import multiprocessing
from src.jobs import main_workflow
from glob import glob
import sys
import os
import time
from src.utils import Config
from src.htcondor_helper import HTCondorHelper
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def initialize_workflow():
    global logger
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_output_dir = os.path.abspath("output")
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp_dir_name = f"output_{timestamp}"
    output_directory = os.path.join(base_output_dir, timestamp_dir_name)
    os.makedirs(output_directory, exist_ok=True)

    # Copy config to output directory
    config_path = os.path.join(output_directory, "config.json")
    shutil.copy("config.json", config_path)
    
    # Initialize config first
    config = Config(config_path)
    logger = config.logger

    logger.info(f"Initializing workflow in {output_directory}")
    return config, output_directory, logger



def organize_output(directory):
    results_dir = os.path.join(directory, "results")
    debug_dir = os.path.join(directory, "debug")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Define patterns for result JSON files
    result_patterns = ["_skim_with_gpt.json", "_km_with_gpt.json", "_km_with_gpt_direct_comp.json"]
    
    for root, dirs, files in os.walk(directory):
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
    start_time = time.time()
    
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
    elif config.job_type == "km_with_gpt_direct_comp":
        terms = [
            {
                "a_term": a_term,
                "b_terms": config.b_terms
            }
            for a_term in config.a_terms
        ]
        logger.debug("TERMS in main:", terms)

    # Maintain the parallel execution pattern
    workflow = partial(
        main_workflow,
        output_dir=output_directory,
        config=config
    )

    with multiprocessing.Pool() as p:
        generated_file_paths = p.map(workflow, terms)

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Main workflow completed in {elapsed_time:.2f} seconds.")
    if not config.using_htcondor:
        logger.error("HTCONDOR configuration is required but not found in config file.")
        return

    htcondor_helper = HTCondorHelper(config)

    try:
        # Create src directory in output directory
        output_src_dir = os.path.join(output_directory, "src")  
        os.makedirs(output_src_dir, exist_ok=True)
        
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
        
        # TODO: CONCAT .tsvs
        with open(files_txt_path, "r") as f:
            tsv_files = [os.path.join(output_directory, line.strip()) for line in f if line.strip()]

        if len(tsv_files) > 1:
            logger.info(f"Concatenating {len(tsv_files)} TSV files")

            try: 
                first_filename = os.path.basename(tsv_files[0])
                job_prefix = first_filename.split("_")[0]

                combined_filename = f"{job_prefix}_combined_output_filtered.tsv"
                combined_tsv_path = os.path.join(output_directory, combined_filename)

                # Read and concatenate TSV files
                dataframes = [pd.read_csv(tsv, sep="\t") for tsv in tsv_files]
                combined_df = pd.concat(dataframes, ignore_index=True)

                # Write the combined TSV file
                combined_df.to_csv(combined_tsv_path, sep="\t", index=False)
                logger.info(f"Concatenated TSV file saved at {combined_tsv_path}")

                # Update files.txt to contain only the new combined file
                with open(files_txt_path, "w") as f:
                    f.write(combined_filename + "\n")

                #TODO: Cost estimator

                # Take the POST_N from config.json 300 tokens per abstract. 
                # Take the number of abstracts in the combined tsv file, taking into consideration POST_N (keeping as an upper bound for each intersection) (INTERSECTION SPECIFIC).
                # Get prompt from prompt_library.py and tokenze it and add per row (4 tokens / 3 words) (ROW SPECIFIC)

                # total input tokens * model-specifc input cost/10^6
                # improve modularity of cost estimator (cost_estimator.py)

                # highlight chosen model, and alternative if needed. 

                # skim row = ab + bc + skim_prompt (total if ac = 0)
                # if ac != 0: ac_count + km_prompt (a = a_term, b = c_term)

                # TODO: use API to dump cost price and completion_tokens

                # Cost estimation
                # if config.job_type == "km_with_gpt":
                #     try:
                #         from src.cost_estimator import estimate_input_costs_km
                #         estimate_input_costs_km(config, combined_df, output_directory)
                        
                #         # # Call dump_output_data to get API usage details
                #         # try:
                #         #     from src.cost_estimator import dump_output_data
                #         #     if dump_output_data(config, combined_df, output_directory):
                #         #         logger.info("Successfully retrieved API usage details")
                #         #     else:
                #         #         logger.error("Failed to retrieve API usage details")
                #         # except Exception as e:
                #         #     logger.error(f"Error retrieving API usage details: {str(e)}", exc_info=True)
                #     except Exception as e:
                #         logger.error(f"Error calculating cost estimation: {str(e)}", exc_info=True)
                #         sys.exit(1)
                # elif config.job_type == "skim_with_gpt":
                #     try:
                #         from src.cost_estimator import estimate_input_costs_skim
                #         estimate_input_costs_skim(config, combined_df, output_directory)
                        
                #         # # Call dump_output_data to get API usage details
                #         # try:
                #         #     from src.cost_estimator import dump_output_data
                #         #     if dump_output_data(config, combined_df, output_directory):
                #         #         logger.info("Successfully retrieved API usage details")
                #         #     else:
                #         #         logger.error("Failed to retrieve API usage details")
                #         # except Exception as e:
                #         #     logger.error(f"Error retrieving API usage details: {str(e)}", exc_info=True)
                #     except Exception as e:
                #         logger.error(f"Error calculating cost estimation: {str(e)}", exc_info=True)
                #         sys.exit(1)

                # # Exit after cost estimation if requested
                # logger.info("Cost estimation completed successfully. Exiting as requested.")
                # sys.exit(0)

            except Exception as e:
                logger.error(f"Error concatenating TSV files: {str(e)}", exc_info=True)

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
            logger.info(f"Jobs submitted with cluster ID {cluster_id}")

            # Monitor jobs
            if htcondor_helper.monitor_jobs(cluster_id):
                logger.info("Jobs completed, retrieving output...")
                htcondor_helper.retrieve_output(cluster_id)

            # Process results
            logger.info("Processing results...")
            organize_output(output_directory)
            extract_and_write_scores(output_directory)
            
            # Cleanup
            htcondor_helper.cleanup(cluster_id)
        finally:
            os.chdir(original_dir)
        
        logger.info(f"Analysis complete. Results are in {output_directory}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

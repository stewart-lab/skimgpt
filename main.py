from datetime import datetime
from functools import partial
from glob import glob
from src.eval_JSON_results import extract_and_write_scores
from src.jobs import main_workflow
from src.utils import Config
from src.htcondor_helper import HTCondorHelper
from src.cost_estimator import calculate_total_cost_and_prompt, KMCostEstimator, SkimCostEstimator
import itertools
import multiprocessing
import os
import pandas as pd
import shutil
import sys
import time
from pathlib import Path

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


def write_token_to_file():
    """Write the HTCondor token from environment variable to the token directory"""
    token = os.getenv('HTCONDOR_TOKEN')
    if not token:
        raise ValueError("HTCONDOR_TOKEN environment variable not set")
    
    token_dir = os.getenv('HTCONDOR_TOKEN_DIR', './token/')
    os.makedirs(token_dir, exist_ok=True)
    
    # HTCondor expects token files to follow a specific format
    # First try the default token file name
    token_file = os.path.join(token_dir, 'condor_token')
    
    # Check if the token needs to be formatted in the HTCondor way
    # HTCondor tokens are usually in the format of a JWT with header.payload.signature
    if not token.count('.') >= 2 and 'eyJ' not in token:
        print("WARNING: Token doesn't appear to be in JWT format, it may not work with HTCondor")
    
    with open(token_file, 'w') as f:
        f.write(token)
    
    # Set secure permissions (owner read-only)
    os.chmod(token_file, 0o600)
    
    # Log token file details for debugging
    print(f"Token file written to: {token_file}")
    print(f"Token file exists: {os.path.exists(token_file)}")
    print(f"Token file size: {os.path.getsize(token_file)}")
    print(f"Token file permissions: {oct(os.stat(token_file).st_mode)}")
    
    return token_file


def remove_token_file(token_file):
    """Remove the token file for security"""
    if os.path.exists(token_file):
        os.remove(token_file)


def test_htcondor_connection(config, token_dir):
    """Test HTCondor connection using token authentication"""
    import htcondor2 as htcondor
    print(f"Testing HTCondor connection with token directory: {token_dir}")
    
    # Set token directory
    htcondor.param["SEC_TOKEN_DIRECTORY"] = token_dir
    htcondor.param["SEC_CLIENT_AUTHENTICATION_METHODS"] = "TOKEN"
    htcondor.param["SEC_DEFAULT_AUTHENTICATION_METHODS"] = "TOKEN"
    htcondor.param["SEC_TOKEN_AUTHENTICATION"] = "REQUIRED"
    
    # Enable debugging
    htcondor.enable_debug()
    
    try:
        # Connect to collector
        collector = htcondor.Collector(config.collector_host)
        submit_host = htcondor.classad.quote(config.submit_host)
        
        # Query scheduler daemon
        print(f"Querying for schedd on {config.submit_host}")
        schedd_ads = collector.query(
            htcondor.AdTypes.Schedd,
            constraint=f"Name=?={submit_host}",
            projection=["Name", "MyAddress", "DaemonCoreDutyCycle", "CondorVersion"]
        )
        
        if not schedd_ads:
            print(f"No scheduler found for {config.submit_host}")
            return False
            
        schedd_ad = schedd_ads[0]
        print(f"Found scheduler: {schedd_ad.get('Name', 'Unknown')}")
        
        # Test schedd connection
        schedd = htcondor.Schedd(schedd_ad)
        test_query = schedd.query(constraint="False", projection=["ClusterId"])
        print(f"Schedd connection successful! Got {len(test_query)} results")
        
        # Test credential daemon
        cred_ads = collector.query(
            htcondor.AdTypes.Credd,
            constraint=f'Name == "{config.submit_host}"'
        )
        
        if not cred_ads:
            print(f"No credential daemon found for {config.submit_host}")
        else:
            cred_ad = cred_ads[0]
            print(f"Found credential daemon: {cred_ad.get('Name', 'Unknown')}")
            credd = htcondor.Credd(cred_ad)
            
            # Test adding credentials
            try:
                credd.add_user_service_cred(htcondor.CredType.OAuth, b"", "rdrive")
                print("Successfully added credential for rdrive")
            except Exception as e:
                print(f"Failed to add credential: {e}")
        
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


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
        
        # Test connection before proceeding
        if not test_htcondor_connection(config, token_dir):
            logger.error("HTCondor connection test failed. Check token and network access.")
            remove_token_file(token_file)
            return
        
        htcondor_helper = HTCondorHelper(config, token_dir)

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
            else:
                logger.info("Single TSV file found, no concatenation needed")
                combined_tsv_path = tsv_files[0]
                combined_df = pd.read_csv(combined_tsv_path, sep="\t")

            # Update files.txt to contain only the combined/single file
            with open(files_txt_path, "w") as f:
                f.write(os.path.basename(combined_tsv_path) + "\n")

            # Cost estimation - moved outside the len(tsv_files) > 1 condition
            logger.info(f"Current job type: {config.job_type}")
            logger.info("Attempting cost estimation...")

            if config.job_type in ["km_with_gpt", "km_with_gpt_direct_comp"]:
                try:
                    estimator = KMCostEstimator(config)
                    input_tokens = estimator.estimate_input_costs(combined_df)
                    
                    if not calculate_total_cost_and_prompt(config, input_tokens):
                        logger.info("Job aborted by user")
                        sys.exit(0)
                    
                except Exception as e:
                    logger.error(f"Error calculating cost estimation: {str(e)}", exc_info=True)
                    sys.exit(1)
            elif config.job_type == "skim_with_gpt":
                try:
                    estimator = SkimCostEstimator(config)
                    input_tokens = estimator.estimate_input_costs(combined_df)
                    
                    if not calculate_total_cost_and_prompt(config, input_tokens):
                        logger.info("Job aborted by user")
                        sys.exit(0)
                    
                except Exception as e:
                    logger.error(f"Error calculating cost estimation: {str(e)}", exc_info=True)
                    sys.exit(1)
            else:
                logger.info(f"Skipping KM cost estimation for job type: {config.job_type}")

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
            logger.info(f"Jobs submitted with cluster ID {cluster_id}")

            # Monitor jobs with enhanced error handling
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

from __future__ import annotations
import ast
import pandas as pd
import os
import shutil
import socket
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from src.utils import Config, RaggedTensor
from src.pubmed_fetcher import PubMedFetcher
from src.triton_client import TritonClient
from src.relevance_helper import (
    getHypothesis, getPrompts,
    postProcess, process_dataframe, process_results,
)


class TritonBatchFailureError(RuntimeError):
    """Raised when all requests in a Triton batch fail, indicating the server is down or unreachable."""
    pass


def gen(
    prompts: RaggedTensor, model: TritonClient, logger=None,
    max_workers: int = None, show_progress: bool = False, batch_chunk_size: int = None
) -> RaggedTensor:
    """Generate outputs using Triton client batch inference with configurable parameters.
    
    Args:
        prompts: RaggedTensor containing input prompts
        model: TritonClient instance for inference (with pre-configured sampling parameters)
        logger: Logger instance for debugging
        max_workers: Maximum concurrent requests (default: None = use client default)
        show_progress: Show progress bar during inference (default: False)
        batch_chunk_size: Process large batches in chunks (default: None = no chunking)
    
    Returns:
        RaggedTensor containing model outputs
    """
    # Debug: log first few prompts
    if logger:
        logger.info(f"DEBUG gen(): Number of prompts: {len(prompts.data)}")
        if len(prompts.data) > 0:
            logger.info(f"DEBUG gen(): First prompt (truncated): {prompts.data[0][:300]}...")
            logger.info(f"DEBUG gen(): Model sampling params - temperature: {model.temperature}, "
                       f"top_p: {model.top_p}, max_tokens: {model.max_tokens}")
        if max_workers:
            logger.info(f"DEBUG gen(): Using max_workers={max_workers}")
        if batch_chunk_size:
            logger.info(f"DEBUG gen(): Using batch_chunk_size={batch_chunk_size}")
    
    batch_results = model.generate_batch(
        text_inputs=prompts.data,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size
    )
    
    # Check for errors and raise on total batch failure
    error_count = sum(1 for r in batch_results if "error" in r)
    if logger:
        logger.info(f"DEBUG gen(): Number of results: {len(batch_results)}")
        if error_count > 0:
            logger.warning(f"DEBUG gen(): {error_count}/{len(batch_results)} requests failed")
        if len(batch_results) > 0:
            logger.info(f"DEBUG gen(): First result: {batch_results[0]}")

    if batch_results and error_count == len(batch_results):
        raise TritonBatchFailureError(
            f"All {len(batch_results)} Triton requests failed — server appears down or unreachable"
        )
    
    outputs = RaggedTensor(
        [result.get("text_output", "") for result in batch_results], 
        prompts.break_point
    )
    return outputs



def run_relevance_analysis(config: Config, km_output_path: str) -> None:
    """Run relevance analysis on the provided TSV file.
    
    Args:
        config: Initialized Config object
        km_output_path: Path to the TSV file to process
    """
    
    logger = config.logger
    logger.debug(f"config: {config}")
    logger.debug(f"km_output_path: {km_output_path}")
    config.load_km_output(km_output_path)   
    start_time = time.time()
    logger.info("Starting relevance analysis...")
    
    try:
        # DNS resolution test in Python
        host = "eutils.ncbi.nlm.nih.gov"
        ip_address = socket.gethostbyname(host)
        logger.debug(f"Python DNS resolution test: Successfully resolved '{host}' to '{ip_address}'")
    except socket.gaierror as e:
        logger.error(f"Python DNS resolution test: Failed to resolve '{host}'. Error: {e}")
        raise  # Re-raise the exception to stop script execution

    out_df = config.data.copy(deep=True)
    logger.debug(f"Working with dataframe of shape {out_df.shape}")

    # Initialize PubMedFetcher
    pubmed_fetcher = PubMedFetcher(
        config=config,
        email="jfreeman@morgridge.org",
        api_key=config.secrets["PUBMED_API_KEY"],
        max_retries=config.max_retries,
        backoff_factor=0.5
    )
    logger.info("Initialized PubMedFetcher")
    
    # Process each row individually (unified for DCH and non-DCH)
    ab_pmids = []
    ab_hypotheses = []

    for _, row in config.data.iterrows():
        a_term = row['a_term']
        logger.debug(f"Row b_term from dataframe: {row['b_term']}, type: {type(row['b_term'])}")
        b_term = row['b_term']

        # Convert string representation of list to actual list
        pmids = ast.literal_eval(row['ab_pmid_intersection'])
        ab_pmids.append(pmids)

        # Generate hypothesis for this specific pair
        # Preserve pipes here; do not strip before relevance
        hypothesis = getHypothesis(config=config, a_term=a_term, b_term=b_term)
        ab_hypotheses.append(hypothesis)

    # Convert to RaggedTensor format
    ab_pmids = RaggedTensor(ab_pmids)
    ab_hypotheses = RaggedTensor(ab_hypotheses)
    all_pmids = ab_pmids.flatten()
    all_hypotheses = ab_hypotheses.expand(ab_pmids.shape)

    if config.is_skim_with_gpt:
        # Process BC and AC terms row by row
        bc_pmids = []
        bc_hypotheses = []
        ac_pmids = []
        ac_hypotheses = []

        for _, row in config.data.iterrows():
            b_term = row['b_term']
            c_term = row['c_term']
            a_term = row['a_term']
            
            # Process BC terms
            bc_pmid_list = ast.literal_eval(row['bc_pmid_intersection'])
            bc_pmids.append(bc_pmid_list)
            bc_hypothesis = getHypothesis(config=config, c_term=c_term, b_term=b_term)
            bc_hypotheses.append(bc_hypothesis)
            
            # Process AC terms if available
            if config.has_ac and 'ac_pmid_intersection' in row:
                ac_pmid_list = ast.literal_eval(row['ac_pmid_intersection'])
                ac_pmids.append(ac_pmid_list)
                ac_hypothesis = getHypothesis(config=config, a_term=a_term, c_term=c_term)
                ac_hypotheses.append(ac_hypothesis)

        # Convert to RaggedTensor format and add to all_pmids/hypotheses
        bc_pmids = RaggedTensor(bc_pmids)
        bc_hypotheses = RaggedTensor(bc_hypotheses)
        all_pmids += bc_pmids.flatten()
        all_hypotheses += bc_hypotheses.expand(bc_pmids.shape)

        if config.has_ac and ac_pmids:
            ac_pmids = RaggedTensor(ac_pmids)
            ac_hypotheses = RaggedTensor(ac_hypotheses)
            all_pmids += ac_pmids.flatten()
            all_hypotheses += ac_hypotheses.expand(ac_pmids.shape)

    # Fetch abstracts
    abstract_map = pubmed_fetcher.fetch_abstracts(all_pmids)
    num_abstracts_fetched = len(abstract_map)
    abstracts = all_pmids.map(lambda pmid: abstract_map.get(str(pmid), ""))
    
    # Ensure abstracts remain flattened for prompt generation; DCH merging happens in process_results

    # ── Attempt Triton inference, fall back to CHTC on total failure ─────
    try:
        # Initialize Triton client for inference
        logger.info("Initializing Triton client for remote inference...")
        try:
            model = TritonClient(
                server_url=config.filter_config.get("SERVER_URL"),
                model_name=config.filter_config.get("MODEL_NAME"),
                temperature=config.filter_config["TEMPERATURE"],
                top_p=config.filter_config["TOP_P"],
                max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1
            )

            if not model.check_server_health():
                logger.error(f"Triton server at {model.server_url} is not ready")
                raise TritonBatchFailureError(f"Triton server at {model.server_url} is not responding")

            logger.info(f"Successfully connected to Triton server at {model.server_url}")
            logger.info(f"Using model: {model.model_name}")
        except TritonBatchFailureError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Triton client: {e}")
            raise TritonBatchFailureError(f"Failed to initialize Triton client: {e}") from e

        # Get optional performance tuning parameters from config
        max_workers = config.global_settings.get("TRITON_MAX_WORKERS", None)
        if max_workers:
            max_workers = int(max_workers)
            logger.info(f"Using configured max_workers={max_workers} for Triton inference")

        show_progress = config.global_settings.get("TRITON_SHOW_PROGRESS", False)
        if isinstance(show_progress, str):
            show_progress = show_progress.lower() in ("true", "1", "yes")

        batch_chunk_size = config.global_settings.get("TRITON_BATCH_CHUNK_SIZE", None)
        if batch_chunk_size:
            batch_chunk_size = int(batch_chunk_size)
            logger.info(f"Using batch_chunk_size={batch_chunk_size} for large batches")

        prompts = getPrompts(abstracts, all_hypotheses)
        answers = gen(prompts, model, logger,
                      max_workers=max_workers,
                      show_progress=show_progress,
                      batch_chunk_size=batch_chunk_size)

    except TritonBatchFailureError as e:
        logger.warning(f"Triton inference failed: {e}")
        logger.info("Falling back to CHTC HTCondor job submission...")
        _run_chtc_fallback(config, km_output_path, logger)
        return

    # ── Triton succeeded — process results locally ────────────────────────
    defaults = 3 * [RaggedTensor([])]
    ab_outputs, bc_outputs, ac_outputs, *_ = chain(answers.split(), defaults)
    ab_abstracts, bc_abstracts, ac_abstracts, *_ = chain(abstracts.split(), defaults)

    postProcess(
        config, ab_outputs, ab_abstracts, ab_hypotheses, out_df, "ab", ab_pmids.shape
    )

    if config.is_skim_with_gpt:
        postProcess(
            config, bc_outputs, bc_abstracts, bc_hypotheses, out_df, "bc", bc_pmids.shape
        )
        if config.has_ac:
            postProcess(
                config, ac_outputs, ac_abstracts, ac_hypotheses, out_df, "ac", ac_pmids.shape
            )

    # Skip process_dataframe for DCH mode - sampling handles context window sizing
    if not config.is_dch:
        out_df = process_dataframe(out_df, config, pubmed_fetcher)

    # Save the initial processed dataframe
    initial_output_file = config.debug_tsv_name if config.debug else config.filtered_tsv_name
    out_df.to_csv(initial_output_file, sep="\t")
    logger.info(f"Saved initial processed data to {initial_output_file}")

    # Check if we need to run iterations
    if config.iterations:
        num_iterations = 1
        if isinstance(config.iterations, bool) and config.iterations:
            logger.warning("iterations is set to True but no number specified, defaulting to 1 iteration")
        elif isinstance(config.iterations, int) and config.iterations > 0:
            num_iterations = config.iterations
            logger.info(f"Will perform {num_iterations} iterations of analysis")
        else:
            logger.warning("Invalid iterations config, defaulting to 1 iteration")

        logger.info(f"Setting up for {num_iterations} iterations")
        for i in range(1, num_iterations + 1):
            iteration_dir = os.path.join(config.km_output_dir, f"iteration_{i}")
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir)
                logger.info(f"Created output directory for iteration {i}: {iteration_dir}")

        filtered_df = out_df.copy(deep=True)

        for iteration in range(1, num_iterations + 1):
            iteration_start_time = time.time()
            logger.info(f"Processing iteration {iteration}/{num_iterations}...")
            config.set_iteration(iteration)
            process_results(filtered_df, config, num_abstracts_fetched)
            iteration_elapsed_time = time.time() - iteration_start_time
            logger.info(f"Iteration {iteration} completed in {iteration_elapsed_time:.2f} seconds")

        logger.info(f"All {num_iterations} iterations completed successfully")
    else:
        logger.info("No iterations requested, processing results once")
        config.current_iteration = 0
        process_results(out_df, config, num_abstracts_fetched)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Relevance analysis completed in {elapsed_time:.2f} seconds")


# ── CHTC fallback helpers ─────────────────────────────────────────────────

def _write_token_to_file():
    """Create a per-job token directory and write the HTCondor token into it."""
    token = os.getenv("HTCONDOR_TOKEN")
    if not token:
        raise ValueError("HTCONDOR_TOKEN environment variable not set")

    root_dir = Path.cwd() / "token_dirs"
    root_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    token_dir = root_dir / f"token_{ts}_{os.getpid()}"
    token_dir.mkdir(parents=True, exist_ok=True)

    token_file = token_dir / "condor_token"
    token_file.write_text(token)
    token_file.chmod(0o600)

    return str(token_file)


def _remove_token_file(token_file: str):
    """Recursively remove the job-specific token directory for security."""
    token_dir = Path(token_file).parent
    if token_dir.exists():
        shutil.rmtree(str(token_dir), ignore_errors=True)


def _run_chtc_fallback(config: Config, km_output_path: str, logger) -> None:
    """Submit the relevance analysis as a CHTC HTCondor job (Docker + GPU + vLLM).

    Stages all required files into the output directory and uses HTCondorHelper
    to submit, monitor, retrieve, and clean up the job.
    """
    from src.htcondor_helper import HTCondorHelper

    if not config.using_htcondor:
        raise RuntimeError(
            "Triton inference failed and no HTCONDOR configuration is present in "
            "config.json to fall back to. Add an HTCONDOR section with "
            "collector_host, submit_host, and docker_image."
        )

    output_directory = Path(config.km_output_dir).resolve()
    km_output = Path(km_output_path).resolve()

    token_file = None
    try:
        # -- Token management --------------------------------------------------
        token_file = _write_token_to_file()
        token_dir = str(Path(token_file).parent)
        logger.info("HTCondor token written to token directory")

        # -- Initialize HTCondor connection ------------------------------------
        try:
            htcondor_helper = HTCondorHelper(config, token_dir)
        except Exception as e:
            logger.error(f"Failed to initialize HTCondor helper: {e}")
            raise

        # -- Stage files into output directory ---------------------------------
        src_dir = Path(__file__).resolve().parent  # kmGPT/src/

        output_src_dir = output_directory / "src"
        output_results_dir = output_directory / "output"
        output_src_dir.mkdir(parents=True, exist_ok=True)
        output_results_dir.mkdir(parents=True, exist_ok=True)

        # Copy relevance_chtc.py to the output directory (the CHTC Docker entry point)
        chtc_src = src_dir / "relevance_chtc.py"
        chtc_dst = output_directory / "relevance_chtc.py"
        if chtc_src.exists():
            shutil.copy2(str(chtc_src), str(chtc_dst))
            logger.debug(f"Staged {chtc_src.name}")
        else:
            raise FileNotFoundError(f"Required file {chtc_src} not found")

        # Copy run.sh
        run_sh_src = src_dir / "run.sh"
        run_sh_dst = output_directory / "run.sh"
        if run_sh_src.exists():
            shutil.copy2(str(run_sh_src), str(run_sh_dst))
            logger.debug("Staged run.sh")
        else:
            raise FileNotFoundError(f"Required file {run_sh_src} not found")

        # Copy all src/*.py into output/src/
        for py_file in src_dir.glob("*.py"):
            dst = output_src_dir / py_file.name
            if py_file.resolve() != dst.resolve():
                shutil.copy2(str(py_file), str(dst))

        # Ensure config.json exists in output directory
        config_dst = output_directory / "config.json"
        if not config_dst.exists():
            shutil.copy2(config.job_config_path, str(config_dst))

        # Ensure secrets.json exists in output directory
        secrets_dst = output_directory / "secrets.json"
        if not secrets_dst.exists() and hasattr(config, "secrets_path"):
            secrets_src = Path(config.secrets_path)
            if secrets_src.exists():
                shutil.copy2(str(secrets_src), str(secrets_dst))

        # Ensure TSV file is in the output directory
        tsv_dst = output_directory / km_output.name
        if km_output.resolve() != tsv_dst.resolve():
            shutil.copy2(str(km_output), str(tsv_dst))

        # Create files.txt referencing the TSV basename
        files_txt_path = output_directory / "files.txt"
        files_txt_path.write_text(km_output.name + "\n")
        logger.info(f"Created files.txt at {files_txt_path}")

        # -- Submit and monitor ------------------------------------------------
        original_dir = os.getcwd()
        os.chdir(str(output_directory))
        try:
            cluster_id = htcondor_helper.submit_jobs(str(files_txt_path))
            logger.info(f"CHTC fallback: submitted cluster {cluster_id}")

            monitoring_success = False
            try:
                monitoring_success = htcondor_helper.monitor_jobs(cluster_id)
                if monitoring_success:
                    logger.info("CHTC fallback: all jobs completed successfully")
                else:
                    logger.warning("CHTC fallback: monitoring ended with some jobs potentially incomplete")
            except Exception as monitor_err:
                logger.error(f"CHTC fallback: error during monitoring: {monitor_err}")
                try:
                    htcondor_helper.release_held_jobs(cluster_id)
                except Exception:
                    pass

            try:
                logger.info("CHTC fallback: retrieving output files...")
                htcondor_helper.retrieve_output(cluster_id)
                logger.info("CHTC fallback: output files retrieved")
            except Exception as retrieve_err:
                logger.error(f"CHTC fallback: error retrieving output: {retrieve_err}")

            try:
                htcondor_helper.cleanup(cluster_id)
                logger.info(f"CHTC fallback: cleaned up cluster {cluster_id}")
            except Exception as cleanup_err:
                logger.error(f"CHTC fallback: cleanup error: {cleanup_err}")

        finally:
            os.chdir(original_dir)

    finally:
        if token_file:
            _remove_token_file(token_file)
            logger.info("HTCondor token file removed for security")

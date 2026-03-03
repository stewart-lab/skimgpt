from __future__ import annotations
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from src.utils import Config, RaggedTensor
from src.triton_client import TritonClient
from src.relevance_helper import run_relevance_pipeline

logger = logging.getLogger(__name__)


class TritonBatchFailureError(RuntimeError):
    """Raised when all requests in a Triton batch fail, indicating the server is down or unreachable."""
    pass


def gen(
    prompts: RaggedTensor, model: TritonClient,
    max_workers: int = None, show_progress: bool = False, batch_chunk_size: int = None
) -> RaggedTensor:
    """Generate outputs using Triton client batch inference with configurable parameters.

    Args:
        prompts: RaggedTensor containing input prompts
        model: TritonClient instance for inference (with pre-configured sampling parameters)
        max_workers: Maximum concurrent requests (default: None = use client default)
        show_progress: Show progress bar during inference (default: False)
        batch_chunk_size: Process large batches in chunks (default: None = no chunking)

    Returns:
        RaggedTensor containing model outputs
    """
    logger.debug(f"gen(): Number of prompts: {len(prompts.data)}")
    if len(prompts.data) > 0:
        logger.debug(f"gen(): First prompt (truncated): {prompts.data[0][:300]}...")
        logger.debug(f"gen(): Model sampling params - temperature: {model.temperature}, "
                    f"top_p: {model.top_p}, max_tokens: {model.max_tokens}")
    if max_workers:
        logger.debug(f"gen(): Using max_workers={max_workers}")
    if batch_chunk_size:
        logger.debug(f"gen(): Using batch_chunk_size={batch_chunk_size}")

    batch_results = model.generate_batch(
        text_inputs=prompts.data,
        max_workers=max_workers,
        show_progress=show_progress,
        batch_chunk_size=batch_chunk_size
    )

    error_count = sum(1 for r in batch_results if "error" in r)
    logger.debug(f"gen(): Number of results: {len(batch_results)}")
    if error_count > 0:
        logger.warning(f"gen(): {error_count}/{len(batch_results)} requests failed")
    if len(batch_results) > 0:
        logger.debug(f"gen(): First result: {batch_results[0]}")

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

    Initialises a Triton inference client and delegates the rest of the
    orchestration to :func:`run_relevance_pipeline`.  If Triton is
    unreachable (or every request in a batch fails), falls back to a
    CHTC HTCondor GPU job.

    Args:
        config: Initialized Config object
        km_output_path: Path to the TSV file to process
    """
    logger.debug(f"config: {config}")
    logger.debug(f"km_output_path: {km_output_path}")

    # Load before try/except so _run_chtc_fallback can use config.km_output_dir
    config.load_km_output(km_output_path)

    try:
        # Initialize Triton client for inference
        logger.info("Initializing Triton client for remote inference...")
        try:
            model = TritonClient(
                server_url=config.filter_config.get("SERVER_URL"),
                model_name=config.filter_config.get("MODEL_NAME"),
                temperature=config.filter_config["TEMPERATURE"],
                top_p=config.filter_config["TOP_P"],
                max_tokens=config.filter_config["MAX_COT_TOKENS"] if config.debug else 1,
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

        def triton_infer(prompts):
            return gen(
                prompts, model,
                max_workers=max_workers,
                show_progress=show_progress,
                batch_chunk_size=batch_chunk_size,
            )

        run_relevance_pipeline(config, km_output_path, infer=triton_infer)

    except TritonBatchFailureError as e:
        logger.warning(f"Triton inference failed: {e}")
        logger.info("Falling back to CHTC HTCondor job submission...")
        _run_chtc_fallback(config, km_output_path)


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


def _run_chtc_fallback(config: Config, km_output_path: str) -> None:
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
                htcondor_helper.retrieve_with_timeout(cluster_id, timeout=120)
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

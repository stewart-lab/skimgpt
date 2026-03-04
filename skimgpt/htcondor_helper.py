from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter

import htcondor2 as htcondor

from skimgpt.utils import Config

logger = logging.getLogger(__name__)


class HTCondorHelper:
    STATUS_DESC = {
        1: "Idle", 2: "Running", 3: "Removed",
        4: "Completed", 5: "Held", 6: "Transferring Output",
        7: "Suspended"
    }

    _REQUIRED_CONFIG_ATTRS = ('collector_host', 'submit_host', 'docker_image')

    _SECURITY_KEYS = (
        "SEC_CLIENT_AUTHENTICATION",
        "SEC_CLIENT_ENCRYPTION",
        "SEC_CLIENT_INTEGRITY",
    )

    def __init__(self, config: Config, token_dir: str) -> None:
        self.config = config
        self.token_dir = token_dir
        self._validate_config()
        self._configure_htcondor_params()
        self._setup_connection()

    @staticmethod
    def _cluster_constraint(cluster_id: int) -> str:
        """Build the HTCondor constraint expression for a cluster."""
        return f"ClusterId == {cluster_id}"

    def _validate_config(self) -> None:
        """Validate required HTCondor configuration."""
        missing = [attr for attr in self._REQUIRED_CONFIG_ATTRS
                   if not getattr(self.config, attr, None)]
        if missing:
            raise ValueError(f"Missing HTCondor config attributes: {', '.join(missing)}")

    def _configure_htcondor_params(self) -> None:
        """Configure all HTCondor parameters in one place."""
        abs_token_dir = os.path.abspath(self.token_dir)

        params = {
            "TOOL_DEBUG": "D_COMMAND",
            "TOOL_LOG": "/dev/null",
            "FILETRANSFER_DEBUG": "FALSE",
            "MAX_TRANSFER_HISTORY_SIZE": "0",
            "SEC_TOKEN_DIRECTORY": abs_token_dir,
            "SEC_CLIENT_AUTHENTICATION_METHODS": "IDTOKENS",
            "SEC_DEFAULT_AUTHENTICATION_METHODS": "IDTOKENS",
            "SEC_CLIENT_AUTHENTICATION": "REQUIRED",
            "SEC_TOKEN_AUTHENTICATION": "REQUIRED",
            "SEC_CLIENT_ENCRYPTION": "REQUIRED",
            "SEC_CLIENT_INTEGRITY": "REQUIRED",
            "SEC_DEFAULT_ENCRYPTION": "REQUIRED",
            "SEC_DEFAULT_INTEGRITY": "REQUIRED",
        }
        for key, value in params.items():
            htcondor.param[key] = value

        logger.info(
            f"HTCondor parameters configured with token directory: {abs_token_dir}; "
            f"AUTH_METHODS={htcondor.param.get('SEC_CLIENT_AUTHENTICATION_METHODS')}; "
            f"ENCRYPTION={htcondor.param.get('SEC_DEFAULT_ENCRYPTION')}; "
            f"INTEGRITY={htcondor.param.get('SEC_DEFAULT_INTEGRITY')}"
        )

    def _setup_connection(self) -> None:
        """Setup connection to HTCondor submit node."""
        token_file = os.path.join(self.token_dir, 'condor_token')
        if not os.path.exists(token_file):
            raise ValueError(f"Token file not found at {token_file}")

        self.collector = htcondor.Collector(self.config.collector_host)
        submit_host = htcondor.classad.quote(self.config.submit_host)

        # Temporarily relax security to discover schedd (some collectors permit anonymous READ)
        logger.debug(f"Querying for schedd on {self.config.submit_host}")
        saved = {k: htcondor.param.get(k, "REQUIRED") for k in self._SECURITY_KEYS}
        try:
            for k in self._SECURITY_KEYS:
                htcondor.param[k] = "OPTIONAL"
            schedd_ads = self.collector.query(
                htcondor.AdTypes.Schedd,
                constraint=f"Name=?={submit_host}",
                projection=["Name", "MyAddress", "DaemonCoreDutyCycle", "CondorVersion"]
            )
        finally:
            for k, v in saved.items():
                htcondor.param[k] = v

        if not schedd_ads:
            raise ValueError(f"No scheduler found for {self.config.submit_host}")

        schedd_ad = schedd_ads[0]
        logger.debug(
            "Found scheduler: %s (Address=%s, Version=%s)",
            schedd_ad.get('Name', 'Unknown'),
            schedd_ad.get('MyAddress', 'Unknown'),
            schedd_ad.get('CondorVersion', 'Unknown'),
        )
        self.schedd = htcondor.Schedd(schedd_ad)

        # Test connection
        logger.debug("Testing connection to schedd...")
        self.schedd.query(constraint="False", projection=["ClusterId"])
        logger.debug("Successfully connected to schedd")

        # Query credential daemon
        logger.debug(f"Querying for credential daemon on {self.config.submit_host}")
        cred_ads = self.collector.query(
            htcondor.AdTypes.Credd,
            constraint=f'Name == "{self.config.submit_host}"'
        )
        if cred_ads:
            logger.debug(f"Found credential daemon: {cred_ads[0].get('Name', 'Unknown')}")
            self.credd = htcondor.Credd(cred_ads[0])
        else:
            logger.warning(f"No credential daemon found for {self.config.submit_host}. Continuing without it.")
            self.credd = None

        logger.info("Successfully connected to HTCondor submit node")

    def submit_jobs(self, files_txt_path: str) -> int:
        """Submit jobs to HTCondor using files.txt."""
        with open(files_txt_path, 'r') as f:
            file_count = sum(1 for line in f if line.strip())

        logger.debug(f"Submitting jobs for {file_count} files")

        submit_desc = htcondor.Submit({
            "universe": "docker",
            "docker_image": self.config.docker_image,
            "docker_pull_policy": "missing",
            "executable": "./run.sh",
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT",
            "transfer_input_files": ".",
            "output": "output/std_out.out",
            "error": "output/std_err.err",
            "log": "run_$(Cluster).log",
            "request_gpus": "1",
            "request_cpus": "1",
            "request_memory": "15GB",
            "request_disk": "15GB",
            "gpus_minimum_memory": "10GB",
            "requirements": "(CUDACapability >= 8.0)",
            "+WantGPULab": "true",
            "+GPUJobLength": '"short"',
            "+is_resumable": "true",
            "stream_error": "True",
            "stream_output": "True",
            "transfer_output_files": "output",
            "transfer_output_remaps": '""',
        })

        result = self.schedd.submit(submit_desc, spool=True)
        self.schedd.spool(result)
        logger.info(f"Successfully submitted {file_count} jobs in cluster {result.cluster()}")
        return result.cluster()

    def monitor_jobs(self, cluster_id: int, check_interval: int = 30) -> bool:
        """Monitor job progress until all jobs complete."""
        constraint = self._cluster_constraint(cluster_id)

        while True:
            try:
                logger.debug(f"Querying status for cluster {cluster_id}")
                ads = self.schedd.query(
                    constraint=constraint,
                    projection=["ProcID", "JobStatus"]
                )
            except Exception as query_err:
                logger.warning(
                    "Failed to query job status for cluster %s (will retry in %s s): %s",
                    cluster_id, check_interval, query_err, exc_info=True,
                )
                time.sleep(check_interval)
                continue

            if not ads:
                logger.warning(f"No jobs found for cluster {cluster_id}")
                return False

            if all(ad.get("JobStatus") == 4 for ad in ads):
                logger.info(f"All jobs in cluster {cluster_id} completed successfully")
                return True

            status_counts = Counter(ad.get("JobStatus", 0) for ad in ads)
            status_msg = ", ".join(
                f"{self.STATUS_DESC.get(k, f'Unknown({k})')}: {v}"
                for k, v in status_counts.items()
            )
            logger.info(f"Cluster {cluster_id} status: {status_msg}")

            logger.info(f"Attempting to retrieve intermediate output files for cluster {cluster_id}")
            try:
                self.retrieve_with_timeout(cluster_id, timeout=60)
            except Exception as retrieve_err:
                logger.warning(f"Failed to retrieve intermediate output files: {retrieve_err}")

            logger.debug(f"Sleeping for {check_interval} seconds before next check")
            time.sleep(check_interval)

    def retrieve_with_timeout(self, cluster_id: int, timeout: int = 60) -> None:
        """Retrieve output files with a timeout using threading."""
        start = time.time()
        logger.debug(f"Starting retrieve operation at {start}")
        constraint = self._cluster_constraint(cluster_id)

        result: dict = {"success": False, "error": None}

        def _retrieve() -> None:
            try:
                self.schedd.retrieve(constraint)
                result["success"] = True
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=_retrieve, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            logger.warning(f"Retrieve operation timed out after {timeout} seconds")
        elif result["success"]:
            logger.info(f"Successfully retrieved output files for cluster {cluster_id}")
        else:
            raise RuntimeError(f"Retrieve failed: {result['error']}")

        logger.debug(f"Retrieve operation took {time.time() - start:.2f} seconds")

    def cleanup(self, cluster_id: int) -> None:
        """Remove all jobs from the queue."""
        constraint = self._cluster_constraint(cluster_id)
        logger.info(f"Removing all jobs in cluster {cluster_id}")
        remove_result = self.schedd.act(htcondor.JobAction.Remove, constraint)
        logger.info(f"Remove result: {remove_result}")
        logger.info(f"Cleanup completed for cluster {cluster_id}")

    def release_held_jobs(self, cluster_id: int) -> None:
        """Release any held jobs in the specified cluster."""
        constraint = self._cluster_constraint(cluster_id)
        logger.info(f"Releasing any held jobs in cluster {cluster_id}")
        release_result = self.schedd.act(htcondor.JobAction.Release, constraint)
        logger.info(f"Release result: {release_result}")

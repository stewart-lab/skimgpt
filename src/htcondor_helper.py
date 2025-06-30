import htcondor2 as htcondor
import time
import os
import threading
from src.utils import Config


class HTCondorHelper:
    def __init__(self, config: Config, token_dir: str):
        """Initialize with Config instance"""
        self.config = config
        self.logger = config.logger
        self.token_dir = token_dir
        self._validate_config()
        self._configure_htcondor_params()
        self._setup_connection()

    def _validate_config(self):
        """Validate required HTCondor configuration"""
        required_attrs = [
            'collector_host',
            'submit_host',
            'docker_image',
            'request_gpus',
            'request_cpus',
            'request_memory',
            'request_disk'
        ]
        
        missing = [attr for attr in required_attrs if not hasattr(self.config, attr)]
        if missing:
            raise ValueError(f"Missing HTCondor config attributes: {', '.join(missing)}")

    def _configure_htcondor_params(self):
        """Configure all HTCondor parameters in one place"""

        htcondor.param["TOOL_DEBUG"] = "D_COMMAND"
        htcondor.param["TOOL_LOG"] = "/dev/null"
        htcondor.param["FILETRANSFER_DEBUG"] = "FALSE"
        htcondor.param["MAX_TRANSFER_HISTORY_SIZE"] = "0"
        abs_token_dir = os.path.abspath(self.token_dir)
        htcondor.param["SEC_TOKEN_DIRECTORY"] = abs_token_dir
        htcondor.param["SEC_CLIENT_AUTHENTICATION_METHODS"] = "TOKEN"
        htcondor.param["SEC_DEFAULT_AUTHENTICATION_METHODS"] = "TOKEN"
        htcondor.param["SEC_TOKEN_AUTHENTICATION"] = "REQUIRED"
        
        self.logger.info(f"HTCondor parameters configured with token directory: {abs_token_dir}")

    def _parse_memory_size(self, size_str: str) -> int:
        """Parse memory/disk size string like '16G' or '16384' into MB"""
        size_str = str(size_str).strip().lower()
        try:
            if size_str.endswith(("gb", "g")):
                suffix_len = 2 if size_str.endswith("gb") else 1
                return int(float(size_str[:-suffix_len]) * 1024)  # Convert GiB to MiB
            elif size_str.endswith(("mb", "m")):
                suffix_len = 2 if size_str.endswith("mb") else 1
                return int(float(size_str[:-suffix_len]))
            else:
                return int(float(size_str))
        except (ValueError, TypeError):
            return None

    def _safe_to_int(self, value) -> int:
        """Safely convert HTCondor values to int"""
        try:
            return int(value)
        except Exception:
            try:
                return int(str(value))
            except Exception:
                return 0

    def _setup_connection(self):
        """Setup connection to HTCondor submit node"""
        try:
            # Verify token file exists
            token_file = os.path.join(self.token_dir, 'condor_token')
            if not os.path.exists(token_file):
                raise ValueError(f"Token file not found at {token_file}")
            
            # Setup collector
            self.collector = htcondor.Collector(self.config.collector_host)
            submit_host = htcondor.classad.quote(self.config.submit_host)
            
            # Query scheduler daemon
            self.logger.debug(f"Querying for schedd on {self.config.submit_host}")
            schedd_ads = self.collector.query(
                htcondor.AdTypes.Schedd,
                constraint=f"Name=?={submit_host}",
                projection=["Name", "MyAddress", "DaemonCoreDutyCycle", "CondorVersion"]
            )
            
            if not schedd_ads:
                raise ValueError(f"No scheduler found for {self.config.submit_host}")
                
            schedd_ad = schedd_ads[0]
            self.logger.debug(f"Found scheduler: {schedd_ad.get('Name', 'Unknown')}")
            self.schedd = htcondor.Schedd(schedd_ad)
            
            # Test connection to schedd
            self.logger.debug("Testing connection to schedd...")
            try:
                test_query = self.schedd.query(constraint="False", projection=["ClusterId"])
                self.logger.debug(f"Successfully connected to schedd, got {len(test_query)} results")
            except Exception as e:
                self.logger.error(f"Failed to connect to schedd: {str(e)}")
                raise
            
            # Query credential daemon
            self.logger.debug(f"Querying for credential daemon on {self.config.submit_host}")
            cred_ads = self.collector.query(
                htcondor.AdTypes.Credd,
                constraint=f'Name == "{self.config.submit_host}"'
            )
            
            if not cred_ads:
                self.logger.warning(f"No credential daemon found for {self.config.submit_host}. Continuing without it.")
                self.credd = None
            else:
                cred_ad = cred_ads[0]
                self.logger.debug(f"Found credential daemon: {cred_ad.get('Name', 'Unknown')}")
                self.credd = htcondor.Credd(cred_ad)
                
            self.logger.info("Successfully connected to HTCondor submit node")
        except Exception as e:
            self.logger.error(f"Failed to setup HTCondor connection: {e}")
            raise

    def submit_jobs(self, files_txt_path: str) -> int:
        """Submit jobs to HTCondor using files.txt"""
        try:
            abs_files_txt = os.path.abspath(files_txt_path)
            
            # Read the files list
            with open(abs_files_txt, 'r') as f:
                files = [line.strip() for line in f if line.strip()]
            
            self.logger.debug(f"Submitting jobs for {len(files)} files")
            
            # Create submit description
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
                "request_gpus": self.config.request_gpus,
                "request_cpus": self.config.request_cpus,
                "request_memory": self.config.request_memory,
                "request_disk": self.config.request_disk,
                "gpus_minimum_memory": "10G",
                "requirements": "(CUDACapability >= 8.0)",
                "+WantGPULab": "true",
                "+GPUJobLength": '"short"',
                "+is_resumable": "true",
                "stream_error": "True",
                "stream_output": "True",
                "transfer_output_files": "output",
                "transfer_output_remaps": '""',
            })

            # Submit jobs for each file using itemdata
            self.logger.debug("Submitting job batch to HTCondor")
            result = self.schedd.submit(submit_desc, spool=True)
            self.schedd.spool(result)  
            self.logger.info(f"Successfully submitted {len(files)} jobs in cluster {result.cluster()}")
            return result.cluster()

        except Exception as e:
            self.logger.error(f"Failed to submit jobs: {e}")
            raise

    def _log_available_resources(self):
        """Query pool to estimate how many startd slots satisfy the job requirements."""
        try:
            # Parse requirements using helper methods
            req_gpus = int(self.config.request_gpus) if str(self.config.request_gpus).isdigit() else None
            req_cpus = int(self.config.request_cpus) if str(self.config.request_cpus).isdigit() else None
            req_mem = self._parse_memory_size(self.config.request_memory)
            req_disk = self._parse_memory_size(self.config.request_disk)

            # Build constraint parts
            constraint_parts = ["CUDACapability >= 8.0"]
            if req_gpus is not None:
                constraint_parts.append(f"TotalGPUs >= {req_gpus}")
            if req_cpus is not None:
                constraint_parts.append(f"Cpus >= {req_cpus}")
            if req_mem is not None:
                constraint_parts.append(f"Memory >= {req_mem}")
            if req_disk is not None:
                constraint_parts.append(f"Disk >= {req_disk}")

            # Parse GPU memory requirement
            gpu_mem_raw_cfg = "10G"  # Default fallback
            try:
                gpu_mem_raw_cfg = self.config.htcondor_config.get("gpus_minimum_memory", "10G")
            except (AttributeError, TypeError):
                pass

            req_gpu_mem_mb = self._parse_memory_size(gpu_mem_raw_cfg)
            if req_gpu_mem_mb is not None:
                constraint_parts.append(f"CUDAGlobalMemoryMb >= {req_gpu_mem_mb}")

            constraint = " && ".join(constraint_parts)

            # Query for available resources
            ads = self.collector.query(
                htcondor.AdTypes.Startd,
                constraint=constraint,
                projection=[
                    "Name", "TotalGPUs", "State", "Activity",
                    "CUDAGlobalMemoryMb", "CUDAGlobalMemory", "Disk", "Memory"
                ]
            )

            available = len(ads)
            free_ads = [ad for ad in ads if ad.get("State") == "Unclaimed" or ad.get("Activity") == "Idle"]
            busy = available - len(free_ads)

            self.logger.info(
                f"Resource availability – machines meeting requirements ({constraint}): "
                f"{available} (free {len(free_ads)}, busy {busy})"
            )
        except Exception as e:
            self.logger.warning(f"Failed to query resource availability: {e}")

    def monitor_jobs(self, cluster_id: int, check_interval: int = 30) -> bool:
        """Monitor job progress"""
        try:
            while True:
                self.logger.debug(f"Querying status for cluster {cluster_id}")
                ads = self.schedd.query(
                    constraint=f"ClusterId == {cluster_id}",
                    projection=["ProcID", "JobStatus"]
                )
                
                if not ads:
                    self.logger.warning(f"No jobs found for cluster {cluster_id}")
                    return False
                
                # Check if all jobs completed
                if all(ad.get("JobStatus") == 4 for ad in ads):
                    self.logger.info(f"All jobs in cluster {cluster_id} completed successfully")
                    return True
                
                # Log status counts
                status_counts = {}
                for ad in ads:
                    status = ad.get("JobStatus", 0)
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                status_desc = {
                    1: "Idle", 2: "Running", 3: "Removed",
                    4: "Completed", 5: "Held", 6: "Transferring Output",
                    7: "Suspended"
                }
                
                status_msg = ", ".join(f"{status_desc.get(k, f'Unknown({k})')}: {v}" 
                                     for k, v in status_counts.items())
                self.logger.info(f"Cluster {cluster_id} status: {status_msg}")
            
                # Retrieve intermediate output files
                self.logger.info(f"Attempting to retrieve intermediate output files for cluster {cluster_id}")
                try:
                    self._retrieve_with_timeout(cluster_id, timeout=60)
                except Exception as retrieve_err:
                    self.logger.warning(f"Failed to retrieve intermediate output files: {retrieve_err}")
                
                # Log resource availability only while all jobs remain idle
                if ads and all(ad.get("JobStatus") == 1 for ad in ads):
                    self._log_available_resources()
                
                self.logger.debug(f"Sleeping for {check_interval} seconds before next check")
                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error monitoring jobs: {e}", exc_info=True)
            raise

    def _retrieve_with_timeout(self, cluster_id: int, timeout: int = 60):
        """Retrieve output files with timeout using threading"""
        retrieve_start = time.time()
        self.logger.debug(f"Starting retrieve operation at {retrieve_start}")
        
        retrieve_success = [False]
        retrieve_error = [None]
        
        def retrieve_thread():
            try:
                self.schedd.retrieve(f"ClusterId == {cluster_id}")
                retrieve_success[0] = True
            except Exception as e:
                retrieve_error[0] = e
        
        thread = threading.Thread(target=retrieve_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            self.logger.warning(f"Retrieve operation timed out after {timeout} seconds")
        elif retrieve_success[0]:
            self.logger.info(f"Successfully retrieved output files for cluster {cluster_id}")
        else:
            raise Exception(f"Retrieve failed: {retrieve_error[0]}")
        
        retrieve_end = time.time()
        self.logger.debug(f"Retrieve operation took {retrieve_end - retrieve_start:.2f} seconds")

    def retrieve_output(self, cluster_id: int):
        """Retrieve output files from completed jobs"""
        try:
            self._retrieve_with_timeout(cluster_id, timeout=120)  # Longer timeout for final retrieve
        except Exception as e:
            self.logger.error(f"Failed to retrieve output: {e}")
            raise

    def cleanup(self, cluster_id: int):
        """Clean up after job completion by releasing any held jobs and removing all jobs"""
        try:
            # Then remove all jobs from the queue
            self.logger.info(f"Removing all jobs in cluster {cluster_id}")
            remove_result = self.schedd.act(htcondor.JobAction.Remove, f"ClusterId == {cluster_id}")
            self.logger.info(f"Remove result: {remove_result}")
                
            self.logger.info(f"Cleanup completed for cluster {cluster_id}")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise

    def release_held_jobs(self, cluster_id: int):
        """Release any held jobs in the specified cluster"""
        try:
            # Release any held jobs
            self.logger.info(f"Releasing any held jobs in cluster {cluster_id}")
            release_result = self.schedd.act(htcondor.JobAction.Release, f"ClusterId == {cluster_id}")
            self.logger.info(f"Release result: {release_result}")
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to release held jobs: {e}")
            return False 
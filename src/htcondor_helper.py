import htcondor
import time
import os
from src.utils import Config



class HTCondorHelper:
    def __init__(self, config: Config):
        """Initialize with Config instance"""
        self.config = config
        self.logger = config.logger
        self._validate_config()
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

    def _setup_connection(self):
        """Setup connection to HTCondor submit node"""
        try:
            # Disable SciTokens cache directory setting and transfer history
            os.environ['_CONDOR_SCITOKENS_DISABLE_CACHE'] = 'true'
            os.environ['_CONDOR_STATS_HISTORY_FILE'] = '/dev/null'
            
            self.collector = htcondor.Collector(self.config.collector_host)
            test_host = htcondor.classad.quote(self.config.submit_host)
            
            # Setup security manager with token
            with htcondor.SecMan() as sess:
                sess.setToken(htcondor.Token(self.config.secrets["HTCONDOR_TOKEN"]))
                schedd_ad = self.collector.query(
                    htcondor.AdTypes.Schedd,
                    constraint=f"Name=?={test_host}",
                    projection=["Name", "MyAddress", "DaemonCoreDutyCycle"]
                )[0]
                self.schedd = htcondor.Schedd(schedd_ad)
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
            
            # Create submit description
            submit_desc = htcondor.Submit({
                "universe": "docker",
                "docker_image": self.config.docker_image,
                "docker_pull_policy": "missing",
                "executable": "./run.sh",
                "arguments": "$(item_filename)",
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
                "gpus_minimum_memory": "46G",
                "requirements": "(CUDACapability >= 8.0) && (TARGET.GPUs_GlobalMemoryMb > 45373)",
                "+WantGPULab": "true",
                "+GPUJobLength": '"short"',
                "stream_error": "True",
                "stream_output": "True",
                "transfer_output_files": "output",
                "environment": f"KM_OUTPUT_FILE=$(item_filename)",
                "+item_filename": "$(Item)",
            })

            # Submit jobs for each file using itemdata
            with htcondor.SecMan() as sess:
                sess.setToken(htcondor.Token(self.config.secrets["HTCONDOR_TOKEN"]))
                result = self.schedd.submit(submit_desc, count=len(files), itemdata=[
                    {"Item": f} for f in files
                ], spool=True)
                cluster_id = result.cluster()
                
                # Spool the files
                self.schedd.spool(list(submit_desc.jobs(clusterid=cluster_id)))
                
            self.logger.info(f"Successfully submitted {len(files)} jobs in cluster {cluster_id}")
            return cluster_id

        except Exception as e:
            self.logger.error(f"Failed to submit jobs: {e}")
            raise

    def monitor_jobs(self, cluster_id: int, check_interval: int = 30) -> bool:
        """Monitor job progress"""
        try:
            while True:
                # Create a new security session for each query
                with htcondor.SecMan() as sess:
                    sess.setToken(htcondor.Token(self.config.secrets["HTCONDOR_TOKEN"]))
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
                
                # Retrieve output files every minute - in a separate security session
                self.logger.info(f"Attempting to retrieve intermediate output files for cluster {cluster_id}")
                try:
                    # Add timeout to prevent hanging
                    retrieve_start = time.time()
                    self.logger.debug(f"Starting retrieve operation at {retrieve_start}")
                    
                    # Set a timeout for the retrieve operation
                    max_retrieve_time = 60  # 60 seconds timeout
                    
                    # Use a separate thread for the retrieve operation
                    import threading
                    retrieve_success = [False]
                    retrieve_error = [None]
                    
                    def retrieve_thread():
                        try:
                            # Create a new security session specifically for retrieve
                            with htcondor.SecMan() as retrieve_sess:
                                retrieve_sess.setToken(htcondor.Token(self.config.secrets["HTCONDOR_TOKEN"]))
                                # Create a new schedd connection for retrieve
                                schedd_ad = self.collector.query(
                                    htcondor.AdTypes.Schedd,
                                    constraint=f"Name=?={htcondor.classad.quote(self.config.submit_host)}",
                                    projection=["Name", "MyAddress"]
                                )[0]
                                retrieve_schedd = htcondor.Schedd(schedd_ad)
                                retrieve_schedd.retrieve(f"ClusterId == {cluster_id}")
                            retrieve_success[0] = True
                        except Exception as e:
                            retrieve_error[0] = e
                    
                    thread = threading.Thread(target=retrieve_thread)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait for the thread with timeout
                    thread.join(max_retrieve_time)
                    
                    if thread.is_alive():
                        self.logger.warning(f"Retrieve operation timed out after {max_retrieve_time} seconds")
                        # Continue with monitoring even if retrieve times out
                    elif retrieve_success[0]:
                        self.logger.info(f"Successfully retrieved intermediate output files for cluster {cluster_id}")
                    else:
                        self.logger.warning(f"Failed to retrieve intermediate output files: {retrieve_error[0]}")
                        
                    retrieve_end = time.time()
                    self.logger.debug(f"Retrieve operation took {retrieve_end - retrieve_start:.2f} seconds")
                    
                except Exception as retrieve_err:
                    self.logger.warning(f"Warning: Failed to retrieve intermediate output files: {retrieve_err}")
                
                self.logger.debug(f"Sleeping for {check_interval} seconds before next check")
                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error monitoring jobs: {e}", exc_info=True)
            raise

    def retrieve_output(self, cluster_id: int):
        """Retrieve output files from completed jobs"""
        try:
            with htcondor.SecMan() as sess:
                sess.setToken(htcondor.Token(self.config.secrets["HTCONDOR_TOKEN"]))
                self.schedd.retrieve(f"ClusterId == {cluster_id}")
            self.logger.info(f"Successfully retrieved output for cluster {cluster_id}")
        except Exception as e:
            self.logger.error(f"Failed to retrieve output: {e}")
            raise

    def cleanup(self, cluster_id: int):
        """Clean up after job completion"""
        try:
            self.logger.info(f"Cleanup completed for cluster {cluster_id}")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise 
import os
import time
import subprocess
import logging
import hashlib
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SSHHelper:
    def __init__(self, ssh_config):
        self.ssh_config = ssh_config
        self.control_path, self.server, self.port, self.username = (
            self._setup_ssh_connection()
        )

    def _setup_ssh_connection(self):
        connections_dir = os.path.expanduser("~/.ssh/connections")
        os.makedirs(connections_dir, exist_ok=True)
        control_path = f"{connections_dir}/{self.ssh_config.get('chtc_username')}@{self.ssh_config.get('server')}:{self.ssh_config.get('port', 22)}"
        if not self._check_persistent_connection(control_path):
            print("Creating new SSH connection...")
            control_path, server, port, username = self._create_persistent_connection(
                self.ssh_config.get("server"),
                self.ssh_config.get("port", 22),
                self.ssh_config.get("chtc_username"),
                self.ssh_config.get("key_path"),
            )
        else:
            print("Reusing existing SSH connection.")
            server = self.ssh_config.get("server")
            port = self.ssh_config.get("port", 22)
            username = self.ssh_config.get("chtc_username")

        return control_path, server, port, username

    def prepare_remote_directories(self, remote_src_path, remote_subdir_path):
        self.execute_command(f"mkdir -p {remote_src_path}", silent=True)
        self.execute_command(f"mkdir -p {remote_subdir_path}", silent=True)

    def transfer_files_to_remote(
        self, output_directory, remote_subdir_path, generated_file_paths
    ):
        remote_file_paths = []
        dynamic_file_names = []

        for path_item in generated_file_paths:
            if not isinstance(path_item, list):
                path_item = [path_item]
            for file_path in path_item:
                local_file = os.path.abspath(file_path)
                file_name = os.path.basename(file_path)
                safe_file_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", file_name)
                remote_file_path = os.path.join(remote_subdir_path, safe_file_name)
                remote_file_paths.append(remote_file_path.split("/")[-1])
                dynamic_file_names.append(f"filtered_{safe_file_name}")
                self.transfer_files(local_file, remote_file_path, silent=True)

        # Transfer the entire src directory
        self.transfer_files(
            self.ssh_config["src_path"],
            os.path.join(self.ssh_config["remote_path"], "src"),
            action="upload_directory",
            silent=True,
        )

        # Transfer config and files.txt
        config_path = os.path.join(output_directory, "config.json")
        files_path = os.path.join(output_directory, "files.txt")

        self.transfer_files(
            config_path,
            os.path.join(remote_subdir_path, "config.json"),
            silent=True,
        )

        with open(files_path, "w+") as f:
            for remote_file in remote_file_paths:
                f.write(f"{remote_file}\n")

        self.transfer_files(files_path, remote_subdir_path, silent=True)

        return remote_file_paths, dynamic_file_names

    def setup_and_submit_job(self, remote_src_path, remote_subdir_path):
        for file in ["run.sub", "run.sh"]:
            check_cmd = f"test -f {remote_src_path}/{file} && echo 'exists' || echo 'not exists'"
            stdout, _ = self.execute_command(check_cmd, silent=True)
            if "exists" in stdout:
                self.execute_command(
                    f"cp {remote_src_path}/{file} {remote_subdir_path}", silent=True
                )
            else:
                logging.warning(f"{file} not found in {remote_src_path}")
                return False  # Indicate that job submission failed

        try:
            stdout, _ = self.execute_command(
                f"cd {remote_subdir_path} && condor_submit run.sub", silent=True
            )
            logging.info(f"Job submitted: {stdout.strip()}")
            return True  # Indicate successful job submission
        except Exception as e:
            logging.error(f"Error submitting job: {str(e)}")
            return False  # Indicate that job submission failed

    def cleanup_remote_directories(self, remote_src_path, remote_subdir_path):
        self.execute_command(f"rm -rf {remote_src_path}", silent=True)
        self.execute_command(f"rm -rf {remote_subdir_path}", silent=True)

    @staticmethod
    def get_file_hash(file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def monitor_files_and_extensions(
        self,
        remote_subdir_path,
        local_output_directory,
        file_names,
        extensions,
        expected_file_count,
        interval=10,
    ):
        err_out_count = 0
        job_terminated = False
        expected_termination_count = expected_file_count * 2
        total_files_transferred = 0
        last_summary_time = time.time()
        file_hashes = {}

        os.makedirs(local_output_directory, exist_ok=True)
        while not job_terminated or err_out_count < expected_file_count:
            new_files_found = False
            for extension in extensions:
                command = f"find {remote_subdir_path} -type f -name '*{extension}'"
                stdout, _ = self.execute_command(command, silent=True)
                for remote_file_path in stdout.strip().split("\n"):
                    if remote_file_path:
                        file_name = os.path.basename(remote_file_path)
                        local_file_path = os.path.join(
                            local_output_directory, file_name
                        )

                        try:
                            self.transfer_files(
                                remote_file_path,
                                local_file_path,
                                action="download",
                                silent=True,
                            )

                            new_hash = self.get_file_hash(local_file_path)
                            if (
                                file_name not in file_hashes
                                or new_hash != file_hashes[file_name]
                            ):
                                file_hashes[file_name] = new_hash
                                new_files_found = True
                                if file_name not in file_hashes:
                                    total_files_transferred += 1
                                logging.info(f"Updated: {file_name}")

                                if extension in [".err", ".out"]:
                                    err_out_count += 1

                                if extension == ".log":
                                    with open(local_file_path, "r") as log_file:
                                        content = log_file.read()
                                        occurrences = content.count("Job terminated")
                                        if occurrences >= expected_termination_count:
                                            job_terminated = True

                                    if self.check_and_handle_transfer_error(
                                        local_file_path
                                    ):
                                        print("Attempting to resubmit the job...")
                                        if self.setup_and_submit_job(
                                            remote_subdir_path, remote_subdir_path
                                        ):
                                            err_out_count = 0  # Reset the count
                                            job_terminated = (
                                                False  # Reset job termination flag
                                            )
                                        else:
                                            print(
                                                "Failed to resubmit the job. Continuing monitoring..."
                                            )
                        except Exception as e:
                            logging.error(
                                f"Error downloading file {remote_file_path}: {str(e)}"
                            )

            for file_name in file_names:
                remote_file_path = os.path.join(remote_subdir_path, file_name)
                local_file_path = os.path.join(local_output_directory, file_name)
                command = f"if [ -f {remote_file_path} ]; then echo 'exists'; else echo 'not exists'; fi"
                output, _ = self.execute_command(command, silent=True)

                if output.strip() == "exists":
                    try:
                        self.transfer_files(
                            remote_file_path,
                            local_file_path,
                            action="download",
                            silent=True,
                        )

                        new_hash = self.get_file_hash(local_file_path)
                        if (
                            file_name not in file_hashes
                            or new_hash != file_hashes[file_name]
                        ):
                            file_hashes[file_name] = new_hash
                            new_files_found = True
                            if file_name not in file_hashes:
                                total_files_transferred += 1
                            logging.info(f"Updated: {file_name}")
                    except Exception as e:
                        logging.error(
                            f"Error downloading file {remote_file_path}: {str(e)}"
                        )

            if not new_files_found:
                time.sleep(interval)

            # Provide a summary every 5 minutes
            current_time = time.time()
            if current_time - last_summary_time >= 300:  # 5 minutes in seconds
                last_summary_time = current_time
                logging.info(
                    f"Summary: {total_files_transferred} files transferred. Job terminated: {job_terminated}. Error/Out count: {err_out_count}"
                )

        if job_terminated and err_out_count >= expected_file_count:
            try:
                self.transfer_files(
                    remote_subdir_path,
                    local_output_directory,
                    action="download_directory",
                    silent=True,
                )
                logging.info("All files have been copied to the local directory.")
            except Exception as e:
                logging.error(
                    f"Error downloading directory {remote_subdir_path}: {str(e)}"
                )

        if not (job_terminated and err_out_count >= expected_file_count):
            logging.warning("Monitoring ended without meeting all conditions.")

    def _create_persistent_connection(self, server, port, username, key_file_path):
        connections_dir = os.path.expanduser("~/.ssh/connections")
        os.makedirs(connections_dir, exist_ok=True)

        control_path = f"{connections_dir}/{username}@{server}:{port}"

        if not os.path.exists(control_path):
            subprocess.run(
                [
                    "ssh",
                    "-M",
                    "-N",
                    "-f",
                    "-o",
                    "ControlMaster=auto",
                    "-o",
                    f"ControlPath={control_path}",
                    "-o",
                    "ControlPersist=2h",
                    "-i",
                    key_file_path,
                    "-p",
                    str(port),
                    f"{username}@{server}",
                ],
                check=True,
            )
            print("Persistent SSH connection established.")
        else:
            print("Reusing existing SSH connection.")

        return control_path, server, port, username

    def _check_persistent_connection(self, control_path):
        result = subprocess.run(
            ["ssh", "-O", "check", "-S", control_path, "dummy"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def execute_command(self, command, silent=False):
        result = subprocess.run(
            [
                "ssh",
                "-S",
                self.control_path,
                "-p",
                str(self.port),
                f"{self.username}@{self.server}",
                command,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error_msg = f"Command '{' '.join(result.args)}' failed with exit status {result.returncode}. STDERR: {result.stderr}"
            logging.error(error_msg)
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        if not silent:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print(f"Exit Status: {result.returncode}")
        return result.stdout, result.stderr

    def transfer_files(self, source_path, target_path, action="upload", silent=False):
        try:
            if action in ["download", "download_directory"]:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if action == "upload":
                if os.path.isdir(source_path):
                    subprocess.run(
                        [
                            "rsync",
                            "-avz",
                            "-e",
                            f"ssh -o ControlPath={self.control_path}",
                            source_path,
                            f"{self.username}@{self.server}:{target_path}",
                        ],
                        check=True,
                        capture_output=silent,
                    )
                else:
                    subprocess.run(
                        [
                            "scp",
                            "-o",
                            f"ControlPath={self.control_path}",
                            source_path,
                            f"{self.username}@{self.server}:{target_path}",
                        ],
                        check=True,
                        capture_output=silent,
                    )
            elif action == "download":
                subprocess.run(
                    [
                        "scp",
                        "-o",
                        f"ControlPath={self.control_path}",
                        f"{self.username}@{self.server}:{source_path}",
                        target_path,
                    ],
                    check=True,
                    capture_output=silent,
                )
            elif action == "download_directory":
                subprocess.run(
                    [
                        "scp",
                        "-r",
                        "-o",
                        f"ControlPath={self.control_path}",
                        f"{self.username}@{self.server}:{source_path}",
                        target_path,
                    ],
                    check=True,
                    capture_output=silent,
                )
            elif action == "upload_directory":
                subprocess.run(
                    [
                        "rsync",
                        "-avz",
                        "-e",
                        f"ssh -o ControlPath={self.control_path}",
                        source_path,
                        f"{self.username}@{self.server}:{target_path}",
                    ],
                    check=True,
                    capture_output=silent,
                )
            if not silent:
                print(
                    f"{action.capitalize()} successful: {source_path} -> {target_path}"
                )
        except subprocess.CalledProcessError as e:
            print(f"Failed to {action} file/directory: {e}")
            raise

    def check_and_handle_transfer_error(self, log_file_path):
        with open(log_file_path, "r") as log_file:
            log_content = log_file.read()
        if (
            "Failed to transfer files" in log_content
            and "No such file or directory" in log_content
        ):
            print(
                "Detected file transfer error. Attempting to create missing directory..."
            )
            match = re.search(r"/home/.*?/src/", log_content)
            if match:
                missing_dir = match.group()
                try:
                    self.execute_command(f"mkdir -p {missing_dir}", silent=True)
                    print(f"Created directory: {missing_dir}")
                    return True
                except Exception as e:
                    print(f"Failed to create directory: {str(e)}")
                    return False
        return False

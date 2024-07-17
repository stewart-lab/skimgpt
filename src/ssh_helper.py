import paramiko
from scp import SCPClient
import os
import time


def create_ssh_client(server, port, username, key_file_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        mykey = paramiko.RSAKey.from_private_key_file(key_file_path)
        client.connect(hostname=server, port=port, username=username, pkey=mykey)
        print("SSH connection established successfully.")
    except Exception as e:
        print(f"Failed to connect: {e}")

    return client


def transfer_files(ssh_client, source_path, target_path, action="upload"):
    """
    Transfer files between a local and remote machine.

    Parameters:
    - ssh_client: The SSH client object.
    - source_path: The source file path.
    - target_path: The target file path.
    - action: "upload" for uploading files to the remote machine, "download" for downloading files from the remote machine. Defaults to "upload".
    """
    # Ensure the target directory exists (for download) or create it (for upload)
    if action == "download":
        local_dir = os.path.dirname(target_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

    try:
        with SCPClient(ssh_client.get_transport()) as scp:
            if action == "upload":
                print(f"Uploading file from {source_path} to {target_path}...")
                scp.put(source_path, target_path, recursive=True)
                print(f"File uploaded: {source_path} -> {target_path}")
            elif action == "download":
                print(f"Downloading file from {source_path} to {target_path}...")
                scp.get(source_path, target_path)
                print(f"File downloaded: {source_path} -> {target_path}")
            elif action == "download_directory":
                scp.get(source_path, target_path, recursive=True)
                print(
                    f"Downloading directory recursively from {source_path} to {target_path}..."
                )
    except Exception as e:
        print(f"Failed to {action} file: {e}")


def transfer_entire_directory(ssh_client, remote_dir, local_dir):
    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Prepare for directory download: need to handle directory as well
    remote_path_with_wildcard = os.path.join(remote_dir, "*")
    transfer_files(ssh_client, remote_path_with_wildcard, local_dir, action="download")


def execute_remote_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    channel = stdout.channel
    while not channel.exit_status_ready():
        pass  # Optionally add time.sleep(1) here

    stdout_result = stdout.read().decode()
    stderr_result = stderr.read().decode()
    exit_status = channel.recv_exit_status()

    print("STDOUT:", stdout_result)
    print("STDERR:", stderr_result)
    print("Exit Status:", exit_status)

    return stdout_result  # Return the stdout result for further processing


def monitor_files_and_extensions(
    ssh_client,
    remote_subdir_path,
    local_output_directory,
    file_names,
    extensions,
    expected_file_count,
    interval=10,
):
    err_out_count = 0  # Counter for .err and .out files
    job_terminated = (
        False  # Flag to check if the termination condition in log files has been met
    )
    expected_termination_count = (
        expected_file_count * 2
    )  # Twice the number of expected_file_count for "Job terminated"

    while not job_terminated or err_out_count < expected_file_count:
        new_files_found = False
        for extension in extensions:
            command = f"find {remote_subdir_path} -type f -name '*{extension}'"
            stdout = execute_remote_command(ssh_client, command).strip().split("\n")
            for remote_file_path in stdout:
                if remote_file_path:
                    file_name = os.path.basename(remote_file_path)
                    local_file_path = os.path.join(local_output_directory, file_name)

                    # Download and overwrite file for continuous updates
                    transfer_files(
                        ssh_client, remote_file_path, local_file_path, action="download"
                    )
                    new_files_found = True

                    if extension in [".err", ".out"]:
                        err_out_count += 1  # Update counts for .err and .out files

                    if extension == ".log":
                        with open(local_file_path, "r") as log_file:
                            content = log_file.read()
                            occurrences = content.count("Job terminated")
                            if occurrences >= expected_termination_count:
                                job_terminated = True

        # Exit the monitoring if both conditions are met
        if job_terminated and err_out_count >= expected_file_count:
            time.sleep(interval)  # Wait for any final files to be written
            transfer_files(
                ssh_client,
                remote_subdir_path,
                local_output_directory,
                action="download_directory",
            )
            print(
                f"All conditions met: {err_out_count} .err/.out files and a single .log file with at least {expected_termination_count} 'Job terminated' strings detected. All files have been copied to the local directory."
            )
            break

        # Check for specified "file_names" not already downloaded in this cycle
        for file_name in file_names:
            remote_file_path = os.path.join(remote_subdir_path, file_name)
            local_file_path = os.path.join(local_output_directory, file_name)
            command = f"if [ -f {remote_file_path} ]; then echo 'exists'; else echo 'not exists'; fi"
            output = execute_remote_command(ssh_client, command).strip()

            if output == "exists":
                transfer_files(
                    ssh_client, remote_file_path, local_file_path, action="download"
                )
                new_files_found = True

        # Reset the cycle tracker after each interval if no new files are found
        if not new_files_found:
            time.sleep(interval)

    # Ensure final condition statement is only reached if the loop ends without both conditions being met
    if not (job_terminated and err_out_count >= expected_file_count):
        print("Monitoring ended without meeting all conditions.")

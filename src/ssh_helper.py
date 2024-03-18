import paramiko
from scp import SCPClient
import os
import time

def create_ssh_client(server, port, username, key_file_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Assuming the private key is not encrypted by a passphrase
        # If it is, use the `password` parameter of `from_private_key_file`
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
    except Exception as e:
        print(f"Failed to {action} file: {e}")

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

def monitor_files_and_extensions(ssh_client, remote_subdir_path, local_output_directory, file_names, extensions, interval=10):
    downloaded_files = set()  # To keep track of which files have been successfully downloaded

    while True:
        all_files_downloaded = True  # Assume all files are downloaded, prove otherwise

        for file_name in file_names:
            if file_name in downloaded_files:
                continue  # Skip checking this file if it's already been downloaded

            all_files_downloaded = False  # Found at least one file not yet downloaded
            
            remote_file_path = os.path.join(remote_subdir_path, file_name)
            local_file_path = os.path.join(local_output_directory, file_name)
            command = f"if [ -f {remote_file_path} ]; then echo 'exists'; else echo 'not exists'; fi"
            output = execute_remote_command(ssh_client, command).strip()

            if output == 'exists':
                transfer_files(ssh_client, remote_file_path, local_file_path, action="download")
                downloaded_files.add(file_name)  # Mark this file as downloaded

        # Check if we've downloaded all specified files
        if all_files_downloaded:
            print("All specified files have been downloaded.")
            break  # Exit the while loop since we've downloaded all specified files

        # Continue fetching files with specified extensions
        for extension in extensions:
            command = f"find {remote_subdir_path} -type f -name '*{extension}'"
            stdout = execute_remote_command(ssh_client, command).strip().split('\n')
            for remote_file_path in stdout:
                if remote_file_path:
                    file_name = os.path.basename(remote_file_path)
                    local_file_path = os.path.join(local_output_directory, file_name)
                    # For files with specified extensions, always fetch and overwrite
                    transfer_files(ssh_client, remote_file_path, local_file_path, action="download")
                    downloaded_files.add(file_name)  # Update as downloaded even if already there to prevent re-downloading

        time.sleep(interval)
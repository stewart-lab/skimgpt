import paramiko
from scp import SCPClient

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

def transfer_files(ssh_client, local_path, remote_path):
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(local_path, remote_path)
        print(f"File transferred: {local_path} -> {remote_path}")
        


def execute_remote_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    stdout_result = stdout.read().decode()
    stderr_result = stderr.read().decode()
    print("STDOUT:", stdout_result)
    print("STDERR:", stderr_result)
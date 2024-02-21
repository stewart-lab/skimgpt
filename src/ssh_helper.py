import paramiko
from scp import SCPClient

def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def transfer_files(ssh_client, local_path, remote_path):
    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(local_path, remote_path)  # For uploading files to remote
        # scp.get(remote_path, local_path)  # For downloading files to local

def execute_remote_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    print(stdout.read().decode())

server = 'your_server_address'
port = 22
user = 'your_username'
password = 'your_password'
executable = 'path/to/your/executable'
config_file = 'path/to/your/config.json'
remote_path = '/remote/directory/'
output_file = 'output_file_name'  # Assume your executable generates this
local_output_path = 'path/to/local/directory/'

# Create SSH client
ssh_client = create_ssh_client(server, port, user, password)

# Transfer files
transfer_files(ssh_client, executable, remote_path + 'executable')
transfer_files(ssh_client, config_file, remote_path + 'config.json')

# Execute the command
execute_remote_command(ssh_client, f'cd {remote_path} && ./executable')

# Download the output file
transfer_files(ssh_client, remote_path + output_file, local_output_path)

# Close SSH connection
ssh_client.close()

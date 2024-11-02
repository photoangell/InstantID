import subprocess
import os
import json

def sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command):
    try:
        # Step 1: Rsync from local to remote
        rsync_command_1 = [
            "rsync", "-avz", "--ignore-existing", "--no-perms", "--no-owner", "--no-group", "-e",
            f"ssh -p {port_number}",
            f"{local_directory}/input", f"root@{ip_address}:{remote_directory}"
        ]
        print(f"Running command: {' '.join(rsync_command_1)}")
        subprocess.run(rsync_command_1, check=True)

        # Step 2: Run a process on the remote machine via SSH
        ssh_command = [
            "ssh", "-p", str(port_number), f"root@{ip_address}",
            remote_command
        ]
        print(f"Running command: {' '.join(ssh_command)}")
        subprocess.run(ssh_command, check=True)

        # Step 3: Rsync from remote back to local
        rsync_command_2 = [
            "rsync", "-avz", "--ignore-existing", "--no-perms", "-e",
            f"ssh -p {port_number}",
            f"root@{ip_address}:{remote_directory}/output", f"{local_directory}"
        ]
        print(f"Running command: {' '.join(rsync_command_2)}")
        subprocess.run(rsync_command_2, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

def load_previous_inputs():
    try:
        with open(os.path.join(os.path.dirname(__file__), "previous_batch_inputs.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_inputs(ip_address, port_number, local_directory):
    with open(os.path.join(os.path.dirname(__file__), "previous_batch_inputs.json"), "w") as f:
        json.dump({
            "ip_address": ip_address,
            "port_number": port_number,
            "local_directory": local_directory
        }, f)

if __name__ == "__main__":
    previous_inputs = load_previous_inputs()

    ip_address = input(f"Enter the IP address of the remote machine [{previous_inputs.get('ip_address', '')}]: ") or previous_inputs.get('ip_address', '')
    port_number = input(f"Enter the port number for SSH [{previous_inputs.get('port_number', '')}]: ") or previous_inputs.get('port_number', '')
    local_directory = input(f"Enter the local directory path [{previous_inputs.get('local_directory', '')}]: ") or previous_inputs.get('local_directory', '')
    remote_directory = "/workspace/img"
    remote_command = "python /workspace/InstantID/gradio_demo/app-multi_batch.py"

    if not os.path.isdir(local_directory):
        print("Error: The specified local directory does not exist.")
    else:
        save_inputs(ip_address, port_number, local_directory)
        
        sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command)

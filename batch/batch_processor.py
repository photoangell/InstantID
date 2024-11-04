import subprocess
import sys
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
        
        print("==============================")
        print(f"Running command: {' '.join(rsync_command_1)}")
        print("==============================")
        subprocess.run(rsync_command_1, check=True)

        # Step 2: Run a process on the remote machine via SSH
        ssh_command = [
            "ssh", "-p", str(port_number), f"root@{ip_address}",
            remote_command
        ]
        
        print("==============================")
        print(f"Running command: {' '.join(ssh_command)}")
        print("==============================")
        try:
            # Run the command, check=True will cause an exception if exit code is not 0
            subprocess.run(ssh_command, check=True)
            print("Script ran successfully.")
        except subprocess.CalledProcessError as e:
            # Handle the error if the script exits with a non-zero status
            print(f"Script failed with exit code {e.returncode}")
            sys.exit()

        # Step 3: Rsync from remote back to local
        rsync_command_2 = [
            "rsync", "-avz", "--ignore-existing", "--no-perms", "-e",
            f"ssh -p {port_number}",
            f"root@{ip_address}:{remote_directory}/output", f"{local_directory}"
        ]
        
        print("==============================")
        print(f"Running command: {' '.join(rsync_command_2)}")
        print("==============================")
        subprocess.run(rsync_command_2, check=True)

    except subprocess.CalledProcessError as e:
        print("==============================")
        print(f"An error occurred while running the command: {e}")
        print("==============================")

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
    local_directory = (input(f"Enter the local directory root batch path [{previous_inputs.get('local_directory', '')}]: ") or previous_inputs.get('local_directory', '')).rstrip('/')
    batch_name = input(f"Enter the batch directory name [{previous_inputs.get('batch_name', '')}]: ") or previous_inputs.get('batch_name', '')
    ip_address = input(f"Enter the IP address of the remote machine [{previous_inputs.get('ip_address', '')}]: ") or previous_inputs.get('ip_address', '')
    port_number = input(f"Enter the port number for SSH [{previous_inputs.get('port_number', '')}]: ") or previous_inputs.get('port_number', '')
    remote_directory = "/workspace/img"
    remote_command = "python /workspace/InstantID/gradio_demo/app-multi_batch.py --batch_name " + batch_name

    if not os.path.isdir(local_directory):
        print("==============================")
        print("Error: The specified local directory does not exist.")
        print("==============================")
    else:
        save_inputs(ip_address, port_number, local_directory)
        
        sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command)

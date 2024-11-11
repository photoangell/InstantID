import subprocess
import sys
import os
import json
import threading
import time

def sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command, batch_name):
    try:
        ssh_command_base = f"ssh -i ~/.ssh/vast_ai -p {port_number}" if is_eric else f"ssh -p {port_number}"
        
        ssh_command = ssh_command_base.split() + [f"root@{ip_address}", f"mkdir -p /workspace/img && mkdir -p /workspace/img/{batch_name}/output"]
        
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
        
        # Step 1: Rsync from local to remote
        rsync_command_1 = [
            "rsync", "-avz", "--no-perms", "--no-owner", "--no-group", "-e",
            ssh_command_base,
            f"{local_directory}/{batch_name}/input", f"root@{ip_address}:{remote_directory}/{batch_name}"
        ]
        
        print("==============================")
        print(f"Running command: {' '.join(rsync_command_1)}")
        print("==============================")
        subprocess.run(rsync_command_1, check=True)

        # Step 2: Run a process on the remote machine via SSH
        ssh_command = ssh_command_base.split() + [f"root@{ip_address}", remote_command]
        rsync_command_2 = [
        "rsync", "-avz", "--no-perms", "-e",
        ssh_command_base,
        f"root@{ip_address}:{remote_directory}/{batch_name}/output", f"{local_directory}/{batch_name}"
    ]
            
        print("==============================")
        print(f"Running command: {' '.join(ssh_command)}")
        print("==============================")
        try:
            main_process = do_batch(ssh_command)

            # Start the rsync file syncing in a separate thread with the specified interval
            rsync_thread = threading.Thread(target=download_results, args=(rsync_command_2, main_process))
            rsync_thread.start()

            # Wait for the main process to complete
            main_process.wait()

            # Ensure the rsync thread stops if the main process ends unexpectedly
            rsync_thread.join()
            
            print("Script ran successfully.")
        except subprocess.CalledProcessError as e:
            # Handle the error if the script exits with a non-zero status
            print(f"Script failed with exit code {e.returncode}")
            sys.exit()
    except subprocess.CalledProcessError as e:
        print("==============================")
        print(f"An error occurred while running the command: {e}")
        print("==============================")

def download_results(rsync_command_2, main_process):
    while main_process.poll() is None:  # Check if main_process is still running
        print("==============================")
        print(f"Running command: {' '.join(rsync_command_2)}")
        print("==============================")
        subprocess.run(rsync_command_2)
        time.sleep(15)
    print("Main process has stopped; running final rsync.")
    subprocess.run(rsync_command_2)
    print("Final rsync complete; stopping rsync loop.")
    
def do_batch(ssh_command):
    return subprocess.Popen(ssh_command)

def load_previous_inputs():
    try:
        with open(os.path.join(os.path.dirname(__file__), "previous_batch_inputs.json"), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_inputs(ip_address, port_number, local_directory, batch_name, is_eric):
    with open(os.path.join(os.path.dirname(__file__), "previous_batch_inputs.json"), "w") as f:
        json.dump({
            "ip_address": ip_address,
            "port_number": port_number,
            "local_directory": local_directory,
            "batch_name": batch_name,
            "is_eric": is_eric
        }, f)

if __name__ == "__main__":
    previous_inputs = load_previous_inputs()
    default_value = "y" if previous_inputs.get('is_eric', False) else "n"
    user_input = input(f"Are you Eric? (y/n) [{default_value}]: ") or default_value
    is_eric = user_input.lower() == 'y'
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
        save_inputs(ip_address, port_number, local_directory, batch_name, is_eric)
        
        sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command, batch_name)

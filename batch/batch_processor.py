import subprocess
import os

def sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command):
    try:
        # Step 1: Rsync from local to remote
        rsync_command_1 = [
            "rsync", "-avz", "-e",
            f"ssh -p {port_number}",
            f"{local_directory}/", f"{ip_address}:{remote_directory}"
        ]
        print(f"Running command: {' '.join(rsync_command_1)}")
        subprocess.run(rsync_command_1, check=True)

        # Step 2: Run a process on the remote machine via SSH
        ssh_command = [
            "ssh", "-p", str(port_number), ip_address,
            remote_command
        ]
        print(f"Running command: {' '.join(ssh_command)}")
        subprocess.run(ssh_command, check=True)

        # Step 3: Rsync from remote back to local
        rsync_command_2 = [
            "rsync", "-avz", "-e",
            f"ssh -p {port_number}",
            f"{ip_address}:{remote_directory}/", local_directory
        ]
        print(f"Running command: {' '.join(rsync_command_2)}")
        subprocess.run(rsync_command_2, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")

if __name__ == "__main__":
    # Take inputs from the user
    ip_address = input("Enter the IP address of the remote machine: ")
    port_number = input("Enter the port number for SSH: ")
    local_directory = input("Enter the local directory path: ")
    remote_directory = input("Enter the remote directory path: ")
    remote_command = input("Enter the command to run on the remote machine: ")

    # Validate that local directory exists
    if not os.path.isdir(local_directory):
        print("Error: The specified local directory does not exist.")
    else:
        sync_and_execute(ip_address, port_number, local_directory, remote_directory, remote_command)
import os
import subprocess
from huggingface_hub import hf_hub_download # type: ignore

def download_checkpoint(repo_id, filename, local_dir):
    # Construct the full local file path
    local_file_path = os.path.join(local_dir, filename)
    
    # Check if the file already exists
    if os.path.exists(local_file_path):
        print(f"File {local_file_path} already exists, skipping download.")
    else:
        # Download the file if it doesn't exist
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
        )
        print(f"Downloaded {filename} to {local_dir}")
        
def git_clone_or_pull(repo_url, repo_dir):
    """
    Clones the repository if it does not exist, or performs a git pull if it does.
    
    :param repo_url: URL of the git repository to clone or pull
    :param repo_dir: Directory where the repo should be cloned or updated
    """
    # Check if the directory exists and is a git repository
    if os.path.isdir(repo_dir):
        # If the directory is a git repository, do a pull
        if os.path.isdir(os.path.join(repo_dir, '.git')):
            print(f"Directory {repo_dir} exists. Pulling latest changes.")
            try:
                subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to pull repository: {e}")
        else:
            print(f"Directory {repo_dir} exists but is not a git repository.")
    else:
        # Clone the repository if it doesn't exist
        print(f"Cloning repository into {repo_dir}")
        try:
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}") 
            
# Download ControlNetModel config
download_checkpoint(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints"
)

# Download ControlNetModel diffusion model
download_checkpoint(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints"
)

# Download IP Adapter
download_checkpoint(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir="./checkpoints"
)

# Download LCM Lora weights
download_checkpoint(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="./checkpoints"
)

# Clone the antelopev2 model repository
git_clone_or_pull(
    repo_url="https://huggingface.co/DIAMONIK7777/antelopev2", 
    repo_dir="./models/antelopev2"
)
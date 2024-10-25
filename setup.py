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
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Ensure actual file download, no symlink
        )
        print(f"Downloaded {filename} to {local_dir}")

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

# Ensure the models directory exists
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

# Clone the repository into the models directory
subprocess.run(["git", "clone", "https://huggingface.co/DIAMONIK7777/antelopev2", models_dir], check=True)

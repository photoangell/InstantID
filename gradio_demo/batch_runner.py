import argparse
import json
import sys
from pathlib import Path
from modules.image_pipeline import initialize_pipeline, generate_image, get_torch_device
import gc


def main(batch_name):
    print('Pipeline building...')
    
    # Initialize pipeline
    pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"
    pipe = initialize_pipeline(pretrained_model_name_or_path)

    # Set all parameters that generate_image will use (stored in a Python dictionary)
    batch_config_path = Path("/workspace") / "img" / batch_name / "input" / "batchconfig.json"

    if not batch_config_path.is_file():
        print(f"Error: Configuration file not found at {batch_config_path}. Please ensure the config file exists.")
        sys.exit(1)
        
    try:
        with batch_config_path.open('r') as config_file:
            generate_image_params = json.load(config_file)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON configuration file at {batch_config_path}.")
        print(f"Details: {e}")
        sys.exit(1)
        
    config = {
        "batch_name": batch_name,
        "generate_image_params": generate_image_params
    }

    run_batch(config, pipe)


def run_batch(config, pipe):
    # Read some of the params in config
    batch_name = config['batch_name']
    generate_image_params = config['generate_image_params']
    
    # Set Paths for batch-specific folders
    passenger_batch_dir = Path("/workspace") / "img" / batch_name / "input" / "passenger"
    reference_batch_dir = Path("/workspace") / "img" / batch_name / "input" / "reference"
    output_batch_dir = Path("/workspace") / "img" / batch_name / "output"

    # Step 1: Check and create batch directories if they don't exist
    passenger_batch_dir.mkdir(parents=True, exist_ok=True)
    reference_batch_dir.mkdir(parents=True, exist_ok=True)
    output_batch_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Generate images and save details
    output_info = []  # Store info for the text file

    # Loop through each passenger image
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    passenger_images = sorted([img for img in passenger_batch_dir.glob('*') if img.suffix.lower() in valid_extensions])
    reference_images = sorted([img for img in reference_batch_dir.glob('*') if img.suffix.lower() in valid_extensions])

    # Ensure there are passenger and reference images
    if not passenger_images or not reference_images:
        raise ValueError("Ensure both passenger and reference directories contain images.")

    base_prompt = generate_image_params.get('prompt', '')

    image_index = 1
    for passenger_image in passenger_images:
        if not passenger_image.is_file():
            continue  # Skip if it's not a file
        
        passenger_filename = passenger_image.stem
        
        for reference_image in reference_images:
            if not reference_image.is_file():
                continue  # Skip if it's not a file
            
            reference_filename = reference_image.stem
            print(f"Processing image {image_index}: {passenger_filename} with reference {reference_filename}")
            
            # Generate image using unpacked parameters
            try:
                generated_image, seed_used = generate_image(
                    pipe=pipe,
                    face_image_path=str(passenger_image),
                    pose_image_path=str(reference_image),
                    **generate_image_params
                )

                # Step 3: Save the generated image
                output_image_path = output_batch_dir / f"{passenger_filename}.{reference_filename}.{seed_used}.output.jpg"
                generated_image.save(output_image_path)

                # Step 4: Write log information
                log_file_path = output_batch_dir / f"{passenger_filename}.{reference_filename}.{seed_used}.output.txt"
                with log_file_path.open('w') as log_file:
                    log_file.write(f"Passenger File: {passenger_image}\n")
                    log_file.write(f"Reference File: {reference_image}\n")
                    log_file.write(f"Output File: {output_image_path}\n")
                    log_file.write(f"Seed Used: {seed_used}\n")
                    log_file.write(f"Parameters: {generate_image_params}\n")
                    log_file.write("\n---\n\n")

                image_index += 1

            except Exception as e:
                print(f"Error processing passenger image {passenger_image} with reference image {reference_image}: {e}")


if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Run InstantID batch processing.")
    parser.add_argument('--batch_name', type=str, required=True, help='The batch directory name for processing')

    args = parser.parse_args()
    
    # Pass the arguments to main()
    main(batch_name=args.batch_name)

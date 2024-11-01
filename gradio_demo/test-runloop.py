import os
from pathlib import Path

config = {
    "passenger_dir": "/workspace/img/inputs/passenger",
    "reference_dir": "/workspace/img/inputs/reference",
    "output_dir": "/workspace/img/output",
    "batch_name": "batch1",
    "generate_image_params": {
        "prompt": "",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        "style_name": "",
        "num_steps": 30,
        "identitynet_strength_ratio": 0.8,
        "adapter_strength_ratio": 0.8,
        "guidance_scale": 5,
        "seed": 42,
        "enable_LCM": False,
        "enhance_face_region": True
    }
}



def run_batch(config):
    # read some of the params in config
    passenger_dir = Path(config['passenger_dir'])
    reference_dir = Path(config['reference_dir'])
    output_dir = Path(config['output_dir'])
    batch_name = config['batch_name']
    generate_image_params = config['generate_image_params']
    
    # Set Paths for batch-specific folders
    passenger_batch_dir = passenger_dir / batch_name
    reference_batch_dir = reference_dir / batch_name
    output_batch_dir = output_dir / batch_name

    # Step 1: Check and create batch directories if they don't exist
    passenger_batch_dir.mkdir(parents=True, exist_ok=True)
    reference_batch_dir.mkdir(parents=True, exist_ok=True)
    output_batch_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Generate images and save details
    output_info = []  # Store info for the text file

    # Loop through each passenger image
    passenger_images = sorted(passenger_batch_dir.glob('*'))
    reference_images = sorted(reference_batch_dir.glob('*'))

    # Ensure there are passenger and reference images
    if not passenger_images or not reference_images:
        raise ValueError("Ensure both passenger and reference directories contain images.")

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
                # generated_image = generate_image(
                #     face_image_path=str(passenger_image),
                #     pose_image_path=str(reference_image),
                #     **generate_image_params
                # )

                # # Step 3: Save the generated image
                output_image_path = output_batch_dir / f"{passenger_filename}.{reference_filename}.output.jpg"
                # generated_image.save(output_image_path)

                # Step 4: Write log information
                log_file_path = output_batch_dir / f"{passenger_filename}.{reference_filename}.output.txt"
                with log_file_path.open('w') as log_file:
                    log_file.write(f"Passenger File: {passenger_image}\n")
                    log_file.write(f"Reference File: {reference_image}\n")
                    log_file.write(f"Output File: {output_image_path}\n")
                    log_file.write(f"Parameters: {generate_image_params}\n")
                    log_file.write("\n---\n\n")

                image_index += 1

            except Exception as e:
                print(f"Error processing passenger image {passenger_image} with reference image {reference_image}: {e}")

# if __name__ == "__main__":
#     main()

run_batch(config)
import os
import time
import imagemod  # Import the third script with the image modification function

input_dir = "/data/input"
output_dir = "/data/output"
processed_files = set()

def process_image(image_path):
    # Calls the function from imagemod to process the image
    imagemod.modify_image(image_path, output_dir)

while True:
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_name not in processed_files:
            print(f"Processing new image: {file_path}")
            process_image(file_path)
            processed_files.add(file_name)
    time.sleep(5)  # Polling interval


# ENTRYPOINT ["python", "watcher.py"] in docker file to set this up
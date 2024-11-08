from PIL import Image
import piexif
import os

def process_image(input_path, output_path, max_width=1920, quality=85):
    # Open the image
    with Image.open(input_path) as img:
        # Resize if the width is larger than the max width
        if img.width > max_width:
            # Calculate new height to maintain aspect ratio
            new_height = int(max_width * img.height / img.width)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        # Check the format and save accordingly
        if img.format == "JPEG":
            # Remove EXIF metadata for JPEG
            img.save(output_path, "JPEG", quality=quality, exif=piexif.dump({}))
        elif img.format == "PNG":
            # Save PNG with optimization to reduce file size
            img.save(output_path, "PNG", optimize=True)

        print(f"Processed {input_path} -> {output_path} (Format: {img.format})")

# Directory paths
input_dir = "input_images"
output_dir = "output_images"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each JPEG or PNG file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_image(input_path, output_path)

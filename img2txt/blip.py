from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the image you want to describe
image = Image.open("/workspace/img/input/passenger/batch1/henry.jpg")

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Process and generate a description
inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)
description = processor.decode(output[0], skip_special_tokens=True)

print("Description:", description)
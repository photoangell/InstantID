import gradio as gr
import json
import sys
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
import platform
import base64
from openai import OpenAI

client = OpenAI()
image_buffer = []
new_image = None
lcm_stored_values = {
    True: {"num_steps": 10, "guidance_scale": 0.0},   # Default values when LCM is enabled
    False: {"num_steps": 30, "guidance_scale": 3.5},  # Default values when LCM is disabled
}

def check_platform():
    release = platform.uname().release.lower()
    if "microsoft" in release:
        return True
    if "rpi" in release or "+rpt" in release:
        return True
    return False

pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"

if not check_platform():
    from modules.image_pipeline import initialize_pipeline, generate_image
    pipe = initialize_pipeline(pretrained_model_name_or_path)
else:
    print("Running on WSL/Raspberry Pi; skipping image_pipeline imports.")

sys.path.append('./')
print('Pipeline building...')


def encode_image(image_path):
    """Encodes an image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_gpt(image_path):
    """
    Analyzes an image using GPT-4-Turbo with vision.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: A structured, concise, and naturalistic Stable Diffusion prompt in the format:
        "[age] years old [race] [gender descriptor] person, [head description]"
    """
    if not image_path:
        return "No image provided."

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini", #    "gpt-4-turbo",
        temperature=0.0,  # Reduce randomness for accuracy
        top_p=0.1,  # Further controls randomness (optional)
        max_tokens=100,  # Prevents excessive response length
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze the image and return a description in the exact following format:\n\n"
                            "[age] years old [race] [gender descriptor] MMA fighter, [head description]\n\n"
                            "Guidelines:\n"
                            "- Replace [age] with the best guess in numbers.\n"
                            "- Replace [race] with the most accurate ethnic descriptor based on facial features, skin tone, and hair type.\n"
                            "  - Possible races: Caucasian, East Asian, South Asian, African, Hispanic, Middle Eastern, Native American, mixed race.\n"
                            "- Do not assume race incorrectly; if uncertain, use 'ambiguous ethnicity'.\n"
                            "- If gender is clear, use 'male' or 'female'. If ambiguous, use 'androgynous' or 'non-binary'.\n"
                            "- For hair:\n"
                            "  - If the person has visible hair, describe its length, texture, and color (e.g., 'short wavy brown hair').\n"
                            "  - If the person is bald, replace '[head description]' with 'bald head'.\n"
                            "  - If the person has a shaved head, use 'shaved head'.\n"
                            "- If the head is covered (hat, hood, turban, hijab, helmet, etc.), describe the covering (e.g., 'wearing a black hijab').\n"
                            "- Do not describe emotions, clothing (except head coverings), accessories, or background.\n"
                            "- Keep the response short, structured, and natural for a Stable Diffusion prompt."
                            "- Do not end with a full stop or punctuation."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content  # Extract and return GPT's response

# Gradio Interface
def update_sliders(enable_lcm, num_steps, guidance_scale):
    """Update num_steps and guidance_scale while remembering previous values for each state."""
    # Save the last used values for the previous state before switching
    lcm_stored_values[not enable_lcm]["num_steps"] = num_steps
    lcm_stored_values[not enable_lcm]["guidance_scale"] = guidance_scale

    # Retrieve the stored values for the new state
    return lcm_stored_values[enable_lcm]["num_steps"], lcm_stored_values[enable_lcm]["guidance_scale"]


def process_uploaded_image(image_path):
    """Handles Gradio image upload and passes it to GPT Vision analysis."""
    return analyze_image_with_gpt(image_path)

def update_gallery(new_image):
    """Append the new image to the gallery while keeping only the last 7 images."""
    global image_buffer

    if new_image is not None:
        image_buffer.append(new_image)  # Add new image to the gallery
        if len(image_buffer) > 7:  # Keep only the last 7 images
            image_buffer.pop(0)  # Remove the oldest image

    return list(reversed(image_buffer))  # Reverse to show the latest first

def process_and_update_gallery(*args):
    """Wrapper function to call call_image_process and update gallery."""
    output_image, seeds_string = call_image_process(*args)  # Process image
    updated_gallery = update_gallery(output_image)  # Update gallery with the new image
    return updated_gallery, seeds_string  # Return both the gallery and seed string

def call_image_process(input_image, reference_image, age, gender, race, hair_length, manual_prompt, gptvision_prompt, prompt, negative_prompt, num_steps, guidance_scale, scheduler, identitynet_strength_ratio, adapter_strength_ratio, controlnet_selection, pose_strength, canny_strength, depth_strength, seed, sigma, strength, threshold, selected_tab, enable_lcm):
    
    if check_platform():
        seeds_string = 12567423
        return input_image, seeds_string 
    
    if selected_tab == 2:
        formatted_prompt = gptvision_prompt + ", " + prompt
    else:
        gender_text = "person" if gender == "ambiguous" else gender
        formatted_prompt = manual_prompt.format(age=str(age), gender=gender_text, race=race, hair_length=hair_length) + ", " + prompt
    style_name = ""
    enhance_face_region = True

    images = []
    seeds_used = []
    for _ in range(1):
        [image, seed_used] = generate_image(pipe,
                input_image,
                reference_image,
                formatted_prompt,
                negative_prompt,
                style_name,
                num_steps,
                identitynet_strength_ratio,
                adapter_strength_ratio,
                pose_strength,
                canny_strength,
                depth_strength,
                controlnet_selection,
                guidance_scale,
                seed,
                scheduler,
                enable_lcm,
                enhance_face_region)
        images.append(image) 
        seeds_used.append(seed_used)
        
    seeds_string = ", ".join(map(str, seeds_used)) 
    sharpened_image = fast_unsharp_mask(images[0], sigma, strength, threshold)
    return sharpened_image, seeds_string 

def fast_unsharp_mask(image, sigma=1.2, strength=1.5, threshold=8):
    """
    Applies a fast unsharp mask using OpenCV.
    
    :param image: Input image (Stable Diffusion output - PIL Image)
    :param sigma: Blur intensity
    :param strength: Sharpening strength
    :return: Sharpened image as a PIL Image
    """
    # ✅ Convert from PIL to NumPy if needed
    if isinstance(image, Image.Image):  
        image = np.array(image)  # Convert PIL to NumPy

    # ✅ Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Normalize if needed

    # ✅ Ensure image is in BGR format for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ✅ Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # ✅ Compute sharpened image
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    # ✅ Apply thresholding to preserve soft areas (e.g., skin)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        sharpened[low_contrast_mask] = image[low_contrast_mask]

    # ✅ Convert back to RGB for correct color display
    sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    # ✅ Convert back to PIL Image for compatibility with Stable Diffusion
    return Image.fromarray(sharpened)

# Define input components
MAX_SEED = np.iinfo(np.int32).max

with gr.Blocks() as demo:
    selected_tab = gr.State(2)  
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Step 1: Upload Images")
            input_image = gr.Image(label="Upload Person Image", type="filepath")
            
            #gr.Markdown("## Step 1a: Select reference image (on startup only)")
            #with gr.Accordion(open=False, label="Reference Image"):
            
            gr.Markdown("# Step 2: Select Attributes")
            with gr.Tabs(selected=2):
                with gr.Tab(label="Manual Attributes", id=1) as tab1:
                    age = gr.Slider(label="Age", minimum=18, maximum=80, step=1, value=35)
                    
                    gender = gr.Radio(
                        choices=["male", "female", "androgynous"],
                        label="Gender",
                        value="male"
                    )
                    race = gr.Radio(
                        choices=["white", "black", "latino", "middle eastern", "east asian", "south asian", "indigenous", "mixed"],
                        label="Race",
                        value="white"
                    )
                    hair_length = gr.Radio(
                        choices=["long hair", "shoulder length hair", "short hair", "curly hair", "afro hair", "dreadlocks", "braided hair", "shaved head", "closely cropped hair", "bald", "headscarf", "turban"],
                        label="Hair Length",
                        value="short hair"
                    )
                    manual_prompt = gr.Textbox(label="prompt",
                                    info="Give simple prompt is enough to achieve good face fidelity", 
                                    value="{age} year old {race} {gender} MMA fighter, {hair_length}")
                    
                with gr.Tab(label="GPT Vision Analysis", id=2) as tab2:
                    with gr.Row():
                        gptvision_prompt = gr.Textbox(label="Person Description Prompt",
                                            info="eg 40 years old Caucasian male MMA fighter, short wavy brown hair",
                                            value="",
                                            scale=4)
                        vision_analysis = gr.Button("Analyse Input Image", scale=1)
                
            gr.Markdown("# Step 2a: Additional Prompts and Options")
            prompt = gr.TextArea(label="Person Attributes Prompt",
                                    info="List physical attributes, expression & pose, photo style & lighting", 
                                    value="physically fit, muscular, strong, intense expression, determined, direct eye contact, eyes looking at the camera, realistic, studio-quality photograph, flat lighting, ring light, evenly lit face, soft light, no shadows, beauty dish lighting, ultra-detailed, high contrast, sharp focus")
            negative_prompt = gr.TextArea(
                        label="Negative Prompt",
                        value="(low quality, worst quality, lowres, low resolution, pixelated, grainy, blurry, out of focus:1.2), (text, artifacts:1.2), cartoon, illustration, anime, deformed, distorted, unnatural, exaggerated, (harsh shadows, high contrast shadows, dramatic lighting, low key lighting, underexposed, moody lighting), dark face, (smiling, teeth:2)")
            with gr.Accordion(open=False, label="Advanced Options"):
                enable_lcm = gr.Checkbox(label="Enable LCM", value=False, info="Enable LCM acceleration for faster generation")
                num_steps = gr.Slider(
                        label="Number of sample steps",
                        info="Number of steps to sample from the model. Higher values can improve quality but may take longer to generate. Use 30, or 10 if you enable LCM",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=30
                    )
                guidance_scale = gr.Slider(
                        label="Guidance scale",
                        info="Use 3.5, or 0.0 if you enable LCM",
                        minimum=0.0,
                        maximum=20.0,
                        step=0.1,
                        value=3.5,
                    )
                schedulers = [
                        "DEISMultistepScheduler",
                        #"HeunDiscreteScheduler", too slow
                        "EulerDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler-Karras",
                        "DPMSolverMultistepScheduler-Karras-SDE",
                        #"DDIMInverseScheduler", noise
                        "DDIMParallelScheduler",
                        "DDIMScheduler",
                        #"DPMSolverMultistepInverseScheduler", noise
                        "DPMSolverSinglestepScheduler",
                        #"DPM++ 2M Karras",
                        #"DPM++ SDE Karras",
                        #"Euler a"                 
                    ]
                scheduler = gr.Dropdown(
                        label="Schedulers",
                        choices=schedulers,
                        value="EulerDiscreteScheduler",
                    )
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                with gr.Row():
                    controlnet_selection = gr.CheckboxGroup(
                        ["pose", "canny", "depth"], label="Controlnet", value=["depth"],
                        info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                    )
                    pose_strength = gr.Slider(
                        label="Pose strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.60,
                    )
                    canny_strength = gr.Slider(
                        label="Canny strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    )
                    depth_strength = gr.Slider(
                        label="Depth strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
                    ) 
                seed = gr.Slider(
                        label="Seed",
                        info="use -1 for random seed",
                        minimum=-1,
                        maximum=MAX_SEED,
                        step=1,
                        value=-1,
                    )
                with gr.Row():
                    gr.Markdown("# Unsharp Masking")
                    sigma = gr.Slider(label="Sigma", info="Blur Intensity. Controls the smoothness of the blurred image used for sharpening. Lower values keep fine details; higher values create a more pronounced effect.", minimum=0.8, maximum=3.0, step=0.1, value=1.2)
                    strength = gr.Slider(label="Strength", info="Sharpening Intensity. Controls how much the sharpened image is amplified. Too high values can cause halos or unnatural contrast.", minimum=1, maximum=2.5, step=0.1, value=1.5)
                    threshold = gr.Slider(label="Threshold", info="Detail Preservation. Prevents sharpening in areas with low contrast (e.g., skin, smooth surfaces). Higher values avoid over-sharpening noise but may reduce effect in subtle areas.", minimum=0, maximum=15, step=1, value=5)
                    
                reference_image = gr.Image(label="Upload Reference Image for pose", type="filepath")
                
        with gr.Column():
            gr.Markdown("# Step 3: Analyze")
            submit_btn = gr.Button("Process Image")
            gr.Markdown("# Step 4: Result")
            gr.Markdown("## Generated Images")
            #outputimage = gr.Image(label="Generated Image", format="jpeg")
            seeds_used = gr.Textbox(label="Seed Used")
            #gr.Markdown("## Previous Images")
            previous_images = gr.Gallery(columns=2, format="jpeg", rows=3)

    
    submit_btn.click(
        fn=process_and_update_gallery,
        inputs=[input_image, reference_image, age, gender, race, hair_length, manual_prompt, gptvision_prompt, prompt, negative_prompt, num_steps, guidance_scale, scheduler, identitynet_strength_ratio, adapter_strength_ratio, controlnet_selection, pose_strength, canny_strength, depth_strength, seed, sigma, strength, threshold, selected_tab, enable_lcm],
        outputs=[previous_images, seeds_used]
    )
    
    vision_analysis.click(fn=process_uploaded_image, inputs=input_image, outputs=gptvision_prompt)

    tab1.select(lambda: 1, outputs=selected_tab)
    tab2.select(lambda: 2, outputs=selected_tab)
    enable_lcm.change(update_sliders, inputs=[enable_lcm, num_steps, guidance_scale], outputs=[num_steps, guidance_scale])

if __name__ == "__main__":
    if check_platform():
        demo.launch()
    else:
        demo.launch(share=True)
import gradio as gr
import json
import sys
from pathlib import Path
import os
import numpy as np
import cv2
from PIL import Image
import platform

def is_wsl():
    if "microsoft" in platform.uname().release.lower():
        return True
    return False

pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"

if not is_wsl():
    from modules.image_pipeline import initialize_pipeline, generate_image
    pipe = initialize_pipeline(pretrained_model_name_or_path)
else:
    print("Running on WSL; skipping image_pipeline imports.")

sys.path.append('./')
print('Pipeline building...')

def call_image_process(input_image, reference_image, gender, race, hair_length, prompt, negative_prompt, num_steps, guidance_scale, scheduler, identitynet_strength_ratio, adapter_strength_ratio, controlnet_selection, pose_strength, canny_strength, depth_strength, seed, sigma, strength, threshold):
    gender_text = "person" if gender == "ambiguous" else gender
        
    formatted_prompt = prompt.format(gender=gender_text, race=race, hair_length=hair_length)
    style_name = ""
    enable_LCM = False
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
                enable_LCM,
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
enable_lcm_arg = False

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Step 1: Upload Images")
            input_image = gr.Image(label="Upload Person Image", type="filepath")
            
            #gr.Markdown("## Step 1a: Select reference image (on startup only)")
            #with gr.Accordion(open=False, label="Reference Image"):
                
            gr.Markdown("# Step 2: Select Attributes")
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
            with gr.Accordion(open=False, label="Advanced Options"):
                prompt = gr.Textbox(label="prompt",
                                    info="Give simple prompt is enough to achieve good face fidelity", 
                                    value="{race} {gender}, {hair_length}, realistic, studio quality photograph, physically fit, healthy, serious, tough, determined, clear focus, transparent background, eyes looking at the camera, MMA fighter")
                negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="lowres, low quality, worst quality:1.2), (text:1.2), Cartoon, illustration, drawing, sketch, painting, anime, (blurry:2.0), out of focus, grainy, pixelated, low resolution, deformed, distorted, unnatural, artificial",
                    )

                num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=5 if enable_lcm_arg else 30
                    )
                guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=0.0 if enable_lcm_arg else 7.0,
                    )
                schedulers = [
                        "DEISMultistepScheduler",
                        "HeunDiscreteScheduler",
                        "EulerDiscreteScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverMultistepScheduler-Karras",
                        "DPMSolverMultistepScheduler-Karras-SDE",
                        "DDIMInverseScheduler",
                        "DDIMParallelScheduler",
                        "DDIMScheduler",
                        "DPMSolverMultistepInverseScheduler",
                        "DPMSolverMultistepScheduler",
                        "DPMSolverSinglestepScheduler"                        
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
                        ["pose", "canny", "depth"], label="Controlnet", value=["pose", "depth"],
                        info="Use pose for skeleton inference, canny for edge detection, and depth for depth map estimation. You can try all three to control the generation process"
                    )
                    pose_strength = gr.Slider(
                        label="Pose strength",
                        minimum=0,
                        maximum=1.5,
                        step=0.05,
                        value=0.40,
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
                    gr.Slider(label="Sigma", info="Blur Intensity. Controls the smoothness of the blurred image used for sharpening. Lower values keep fine details; higher values create a more pronounced effect.", minimum=0.8, maximum=3.0, step=0.1, value=1.2)
                    gr.Slider(label="Strength", info="Sharpening Intensity. Controls how much the sharpened image is amplified. Too high values can cause halos or unnatural contrast.", minimum=1, maximum=2.5, step=0.1, value=1.5)
                    gr.Slider(label="Threshold", info="Detail Preservation. Prevents sharpening in areas with low contrast (e.g., skin, smooth surfaces). Higher values avoid over-sharpening noise but may reduce effect in subtle areas.", minimum=0, maximum=15, step=1, value=5)
                    
                reference_image = gr.Image(label="Upload Reference Image for pose", type="filepath")
                
        with gr.Column():
            gr.Markdown("# Step 3: Analyze")
            submit_btn = gr.Button("Process Image")
            gr.Markdown("# Step 4: Choose from Results")
            # gallery = gr.Gallery(label="Generated Images", columns=2, format="jpeg")
            outputimage = gr.Image(label="Generated Image")
            seeds_used = gr.Textbox(label="Seed Used")
    
    
    submit_btn.click(
        fn=call_image_process,
        inputs=[input_image, reference_image, gender, race, hair_length, prompt, negative_prompt, num_steps, guidance_scale, scheduler, identitynet_strength_ratio, adapter_strength_ratio, controlnet_selection, pose_strength, canny_strength, depth_strength, seed, sigma, strength, threshold],
        outputs=[outputimage, seeds_used]
    )

if __name__ == "__main__":
    if is_wsl():
        demo.launch()
    else:
        demo.launch(share=True)
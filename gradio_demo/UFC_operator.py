import gradio as gr
import json
import sys
from pathlib import Path
import os
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

def call_image_process(input_image, reference_image, gender, race, hair_length, prompt):
    
    gender_text = "person" if gender == "ambiguous" else gender
    if hair_length == "none":
        hair_length_text = "bald head"
    else:
        hair_length_text = f"{hair_length} hair"
        
    formatted_prompt = prompt.format(gender=gender_text, race=race, hair_length=hair_length_text)
    negative_prompt= "lowres, low quality, worst quality:1.2), (text:1.2), Cartoon, illustration, drawing, sketch, painting, anime, (blurry:2.0), out of focus, grainy, pixelated, low resolution, deformed, distorted, unnatural, artificial"
    style_name = ""
    num_steps = 30
    identitynet_strength_ratio = 0.8
    adapter_strength_ratio = 0.8
    pose_strength = 0.4
    canny_strength = 0.4
    depth_strength = 0.4
    controlnet_selection = ["pose", "depth"]
    guidance_scale = 5
    seed = -1
    scheduler = "EulerDiscreteScheduler"
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
    return images, seeds_string 

# Define input components
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Step 1: Upload Images")
            input_image = gr.Image(label="Upload Person Image", type="filepath")
            
            gr.Markdown("## Step 1a: Select reference image (on startup only)")
            with gr.Accordion(open=False, label="Reference Image"):
                reference_image = gr.Image(label="Upload Reference Image", type="filepath")
            
            gr.Markdown("# Step 2: Select Attributes")
            gender = gr.Radio(
                choices=["male", "female", "ambiguous"],
                label="Gender",
                value="male"
            )
            race = gr.Radio(
                choices=["white", "black", "arabic", "asian", "oriental"],
                label="Race",
                value="white"
            )
            hair_length = gr.Radio(
                choices=["long", "medium", "short", "none"],
                label="Hair Length",
                value="medium"
            )
            with gr.Accordion(open=False, label="Advanced Options"):
                prompt = gr.Textbox(label="prompt", value="{race} {gender}, {hair_length}, realistic, studio quality photograph, physically fit, healthy, serious, tough, determined, clear focus, transparent background")
                
            gr.Markdown("# Step 3: Analyze")
            submit_btn = gr.Button("Process Image")
        
        with gr.Column():
            gr.Markdown("# Step 4: Choose from Results")
            gallery = gr.Gallery(label="Generated Images", columns=2)
            seeds_used = gr.Textbox(label="Seed Used")
    
    
    submit_btn.click(
        fn=call_image_process,
        inputs=[input_image, reference_image, gender, race, hair_length, prompt],
        outputs=[gallery, seeds_used]
    )

if __name__ == "__main__":
    if is_wsl():
        demo.launch()
    else:
        demo.launch(share=True)
import gradio as gr
import json
import sys
from pathlib import Path
from modules.image_pipeline import initialize_pipeline, generate_image

sys.path.append('./')
print('Pipeline building...')

# Initialize pipeline
pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"
pipe = initialize_pipeline(pretrained_model_name_or_path)

def call_image_process(input_image, reference_image, gender, race, hair_length):
    
    prompt = f"{race} {gender}, {hair_length} hair, realistic, studio quality photograph, physically fit, healthy, serious, tough, determined, clear focus, transparent background"
    negative_prompt= "lowres, low quality, worst quality:1.2), (text:1.2), Cartoon, illustration, drawing, sketch, painting, anime, (blurry:2.0), out of focus, grainy, pixelated, low resolution, deformed, distorted, unnatural, artificial"
    style_name = ""
    num_steps = 8
    identitynet_strength_ratio = 0.8
    adapter_strength_ratio = 0.8
    pose_strength = 0.4
    canny_strength = 0.4
    depth_strength = 0.4
    controlnet_selection = ["pose", "depth"]
    guidance_scale = 0
    seed = 572504474
    scheduler = "EulerDiscreteScheduler"
    enable_LCM = True
    enhance_face_region = True

    
    [image, seed_used] = generate_image(pipe,
            input_image,
            reference_image,
            prompt,
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
     
    return image, seed_used 

# Define input components
with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(label="Upload Person Image", type="filepath")
        reference_image = gr.Image(label="Upload Reference Image", type="filepath")
    
    with gr.Row():
        gender = gr.Radio(
            choices=["male", "female", "not specified"],
            label="Gender",
            value="not specified"
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
    
    output = gr.Textbox(label="Analysis Results")
    with gr.Column():
        gallery = gr.Image(label="Generated Images")
        seed_used = gr.Textbox(label="Seed Used")
                
    submit_btn = gr.Button("Analyze")
    submit_btn.click(
        fn=call_image_process,
        inputs=[input_image, reference_image, gender, race, hair_length],
        outputs=[gallery, seed_used]
    )

if __name__ == "__main__":
    demo.launch(share=True)
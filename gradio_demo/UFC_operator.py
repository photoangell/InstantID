import gradio as gr
import argparse
import json
import sys
from pathlib import Path
from modules.image_pipeline import initialize_pipeline, generate_image

sys.path.append('./')
print('Pipeline building...')

# Initialize pipeline
pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"
pipe = initialize_pipeline(pretrained_model_name_or_path)

def analyze_person(input_image, reference_image, gender, race, hair_length):
    # Here you would add your image analysis logic
    generate_image
    
    return f"Analysis Results:\nGender: {gender}\nRace: {race}\nHair Length: {hair_length}"

# Define input components
with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(label="Upload Person Image")
        reference_image = gr.Image(label="Upload Reference Image")
    
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
                
    submit_btn = gr.Button("Analyze")
    submit_btn.click(
        fn=analyze_person,
        inputs=[input_image, reference_image, gender, race, hair_length],
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch(share=True)
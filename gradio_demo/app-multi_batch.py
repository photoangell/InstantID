### Eric: I'm going to use this version to troubleshoot getting it up and running. 
# rip out styles and examples

# appended rudimentary batch process

import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # or .resolve().parents[n] for n levels up

# Add project root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
print (f"script path is {SCRIPT_DIR}, project root is {PROJECT_ROOT}")

from typing import Tuple
import os
import argparse
import cv2
import math
import torch
import random
import numpy as np
import argparse

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl, get_torch_device, torch_gc
from controlnet_util import openpose, get_depth_map, get_canny_image

import gradio as gr


# global variable
MAX_SEED = np.iinfo(np.int32).max
device = get_torch_device()
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

# Load face encoder
app = FaceAnalysis(
    name="antelopev2",
    root=str(PROJECT_ROOT),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = PROJECT_ROOT / "checkpoints/ip-adapter.bin"
controlnet_path = PROJECT_ROOT / "checkpoints/ControlNetModel"

# Load pipeline face ControlNetModel
controlnet_identitynet = ControlNetModel.from_pretrained(
    controlnet_path, torch_dtype=dtype
)

# controlnet-pose
controlnet_pose_model = "thibaud/controlnet-openpose-sdxl-1.0"
controlnet_canny_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_depth_model = "diffusers/controlnet-depth-sdxl-1.0-small"

controlnet_pose = ControlNetModel.from_pretrained(
    controlnet_pose_model, torch_dtype=dtype
).to(device)
controlnet_canny = ControlNetModel.from_pretrained(
    controlnet_canny_model, torch_dtype=dtype
).to(device)
controlnet_depth = ControlNetModel.from_pretrained(
    controlnet_depth_model, torch_dtype=dtype
).to(device)

controlnet_map = {
    "pose": controlnet_pose,
    "canny": controlnet_canny,
    "depth": controlnet_depth,
}
controlnet_map_fn = {
    "pose": openpose,
    "canny": get_canny_image,
    "depth": get_depth_map,
}

def toggle_lcm_ui(value):
    if value:
        return (
            gr.update(minimum=0, maximum=100, step=1, value=5),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
        )
    else:
        return (
            gr.update(minimum=5, maximum=100, step=1, value=30),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
        )

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def remove_tips():
    return gr.update(visible=False)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(
    image_pil,
    kps,
    color_list=[
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ],
):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def apply_style(
    style_name: str, positive: str, negative: str = ""
) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + " " + negative

def generate_image(
    pipe,
    face_image_path,
    pose_image_path,
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
    enhance_face_region,
    progress=gr.Progress(track_tqdm=True),
):

    if enable_LCM:
        pipe.scheduler = diffusers.LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_lora()
    else:
        pipe.disable_lora()
        scheduler_class_name = scheduler.split("-")[0]

        add_kwargs = {}
        if len(scheduler.split("-")) > 1:
            add_kwargs["use_karras_sigmas"] = True
        if len(scheduler.split("-")) > 2:
            add_kwargs["algorithm_type"] = "sde-dpmsolver++"
        scheduler = getattr(diffusers, scheduler_class_name)
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config, **add_kwargs)

    if face_image_path is None:
        raise gr.Error(
            f"Cannot find any input face image! Please upload the face image"
        )

    if prompt is None:
        prompt = "a person"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side=1024)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error(
            f"Unable to detect a face in the image. Please upload a different photo with a clear face."
        )

    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])
    img_controlnet = face_image
    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image, max_side=1024)
        img_controlnet = pose_image
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(
                f"Cannot find any face in the reference image! Please upload another person image"
            )

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    if len(controlnet_selection) > 0:
        controlnet_scales = {
            "pose": pose_strength,
            "canny": canny_strength,
            "depth": depth_strength,
        }
        pipe.controlnet = MultiControlNetModel(
            [controlnet_identitynet]
            + [controlnet_map[s] for s in controlnet_selection]
        )
        control_scales = [float(identitynet_strength_ratio)] + [
            controlnet_scales[s] for s in controlnet_selection
        ]
        control_images = [face_kps] + [
            controlnet_map_fn[s](img_controlnet).resize((width, height))
            for s in controlnet_selection
        ]
    else:
        pipe.controlnet = controlnet_identitynet
        control_scales = float(identitynet_strength_ratio)
        control_images = face_kps

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_images,
        control_mask=control_mask,
        controlnet_conditioning_scale=control_scales,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images

    return images[0] #, gr.update(visible=True)

def main(batch_name, pretrained_model_name_or_path="wangqixun/YamerMIX_v8", enable_lcm_arg=False):
    print('Pipeline building...')
    
    if pretrained_model_name_or_path.endswith(
        ".ckpt"
    ) or pretrained_model_name_or_path.endswith(".safetensors"):
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            scheduler_name=None,
            weight_dtype=dtype,
        )

        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        pipe = StableDiffusionXLInstantIDPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
            controlnet=[controlnet_identitynet],
        ).to(device)

    else:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            pretrained_model_name_or_path,
            controlnet=[controlnet_identitynet],
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    pipe.load_ip_adapter_instantid(face_adapter)
    # load and disable LCM
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.disable_lora()

    print('Pipeline built...')
    print('Running batch built...')

    # Set all parameters that generate_image will use, (stored in a python dictionary)

    config = {
        "passenger_dir": "/workspace/img/input/passenger",
        "reference_dir": "/workspace/img/input/reference",
        "output_dir": "/workspace/img/output",
        "batch_name": "batch1",
        "generate_image_params": {
            "prompt": "",
            "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            "style_name": "",
            "num_steps": 30,
            "identitynet_strength_ratio": 0.8,
            "adapter_strength_ratio": 0.8,
            "pose_strength": 0.4,
            "canny_strength": 0.3,
            "depth_strength": 0.5,
            "controlnet_selection": ["pose", "canny"],
            "guidance_scale": 5,
            "scheduler": "EulerDiscreteScheduler",
            "seed": 42,
            "enable_LCM": False,
            "enhance_face_region": True
        }
    }

    run_batch(config, pipe)

def run_batch(config, pipe):
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
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    passenger_images = sorted([img for img in passenger_batch_dir.glob('*') if img.suffix.lower() in valid_extensions])
    reference_images = sorted([img for img in reference_batch_dir.glob('*') if img.suffix.lower() in valid_extensions])

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
                generated_image = generate_image(
                    pipe = pipe,
                    face_image_path=str(passenger_image),
                    pose_image_path=str(reference_image),
                    prompt=generate_image_params.get('prompt', ''),
                    negative_prompt=generate_image_params.get('negative_prompt', ''),
                    style_name=generate_image_params.get('style_name', ''),
                    num_steps=generate_image_params.get('num_steps', 30),
                    identitynet_strength_ratio=generate_image_params.get('identitynet_strength_ratio', 0.8),
                    adapter_strength_ratio=generate_image_params.get('adapter_strength_ratio', 0.8),
                    pose_strength=generate_image_params.get('pose_strength', 0.4),
                    canny_strength=generate_image_params.get('canny_strength', 0.3),
                    depth_strength=generate_image_params.get('depth_strength', 0.5),
                    controlnet_selection=generate_image_params.get('controlnet_selection', ["pose", "canny"]),
                    guidance_scale=generate_image_params.get('guidance_scale', 5),
                    scheduler=generate_image_params.get('scheduler', 'EulerDiscreteScheduler'),
                    seed=generate_image_params.get('seed', 42),
                    enable_LCM=generate_image_params.get('enable_LCM', False),
                    enhance_face_region=generate_image_params.get('enhance_face_region', True)
                )

                # Step 3: Save the generated image
                output_image_path = output_batch_dir / f"{passenger_filename}.{reference_filename}.output.jpg"
                generated_image.save(output_image_path)

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

if __name__ == "__main__":
# Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description="Run InstantID batch processing.")
    parser.add_argument('--batch_name', type=str, required=True, help='The batch directory name for processing')

    args = parser.parse_args()
    
    # Pass the arguments to main()
    main(batch_name=args.batch_name)


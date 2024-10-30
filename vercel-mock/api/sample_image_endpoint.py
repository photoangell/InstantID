from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

SAMPLE_IMAGE_PATH = os.path.join(os.getcwd(), "public", "sample_image_1.jpg")

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Define expected parameters with their default values
    expected_params = {
        "face_image_path": None,
        "pose_image_path": None,
        "prompt": "",
        "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        "style_name": "Watercolor",
        "num_steps": 30,
        "identitynet_strength_ratio": 0.8,
        "adapter_strength_ratio": 0.8,
        "guidance_scale": 5,
        "seed": 42,
        "enable_LCM": False,
        "enhance_face_region": True
    }

    # Parse and validate incoming JSON data
    data = request.json
    missing_params = [key for key in expected_params if key not in data]
    
    if missing_params:
        return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400
    
    # Return the sample image and mock usage tips
    # this is what is returned by gradio, but I think it should return an image right?
    # up for discussion
    return jsonify({
        "generated_image_path": SAMPLE_IMAGE_PATH,
        "usage_tips": "This is a mock response. In production, this would contain actual usage tips and the generated image path."
    })

@app.route('/download_image', methods=['GET'])
def download_image():
    """Endpoint to download the sample image."""
    try:
        return send_file(SAMPLE_IMAGE_PATH, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  
from flask import Flask, jsonify, request, send_file
from flask_restx import Api, Resource, fields
import os

app = Flask(__name__)
api = Api(app, title="ymbbt FaceClone Mock API", description="API for generating sample images with Swagger documentation.")

SAMPLE_IMAGE_PATH = os.path.join(os.getcwd(), "public", "sample_image_1.jpg")

# Define the expected request payload with Swagger documentation
generate_image_model = api.model('GenerateImage', {
    'face_image_path': fields.String(required=True, description="Path to the face image file"),
    'pose_image_path': fields.String(required=True, description="Path to the pose image file"),
    'prompt': fields.String(default="", description="Prompt for image generation"),
    'negative_prompt': fields.String(
        default="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, ...",
        description="Negative prompt to avoid specific features"
    ),
    'style_name': fields.String(
        default="Watercolor",
        description="Style for the generated image",
        enum=['(No style)', 'Watercolor', 'Film Noir', 'Neon', 'Jungle', 'Mars', 'Vibrant Color', 'Snow', 'Line art']
    ),
    'num_steps': fields.Float(default=30, description="Number of sample steps"),
    'identitynet_strength_ratio': fields.Float(default=0.8, description="IdentityNet strength for fidelity"),
    'adapter_strength_ratio': fields.Float(default=0.8, description="Image adapter strength for detail"),
    'guidance_scale': fields.Float(default=5, description="Guidance scale"),
    'seed': fields.Float(default=42, description="Seed for reproducibility"),
    'enable_LCM': fields.Boolean(default=False, description="Enable Fast Inference with LCM"),
    'enhance_face_region': fields.Boolean(default=True, description="Enhance the non-face region")
})

# Response model for Swagger documentation
response_model = api.model('GenerateImageResponse', {
    'generated_image_path': fields.String(description="Path to the generated image"),
    'usage_tips': fields.String(description="Usage tips for the generated image")
})

@api.route('/generate_image', methods=['POST'])
class GenerateImage(Resource):
    @api.expect(generate_image_model)
    @api.marshal_with(response_model)
    def post(self):
        """Generate a sample image (mock response)."""
        data = request.json

        # Check for required parameters
        for key in generate_image_model.keys():
            if key not in data:
                return {"error": f"Missing parameter: {key}"}, 400

        # Mock response, no actual image processing
        return {
            "generated_image_path": SAMPLE_IMAGE_PATH,
            "usage_tips": "This is a mock response. In production, this would contain actual usage tips and the generated image path."
        }
        
@app.route('/download_image', methods=['GET'])
def download_image():
    """Endpoint to download the sample image."""
    try:
        return send_file(SAMPLE_IMAGE_PATH, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add Swagger UI at the root
@app.route('/')
def swagger_ui():
    return jsonify(api.__schema__)  # Exposes the Swagger JSON schema directly

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

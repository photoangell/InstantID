from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/api/image')
def serve_image():
    return send_file(os.path.join(os.getcwd(), "public", "sample_image_1.jpg"), mimetype='image/jpeg')

from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
import base64
from PIL import Image
from call_model import superscale_image
from resnet import use_model

import torch
import tempfile
import os

app = Flask(__name__)

# Path to your ResNet model weights
RESNET_WEIGHTS_PATH = 'resnet/model_weigths.pth'  # <-- update as needed


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/superscale', methods=['POST'])
def superscale():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    sr_image = superscale_image(image)

    # Save the processed image to a temporary file for ResNet input
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        sr_image.save(tmp, 'PNG')
        tmp_path = tmp.name

    try:
        # Load ResNet model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = use_model.load_model(RESNET_WEIGHTS_PATH, device=device)
        # Preprocess the image for ResNet
        image_tensor = use_model.preprocess_image(tmp_path)
        # Predict
        class_idx, class_name = use_model.predict(model, image_tensor, device)
    finally:
        # Clean up temp file
        os.remove(tmp_path)

    # Convert the super resolution image to base64 for JSON response
    img_buffer = io.BytesIO()
    sr_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return jsonify({
        'predicted_class_name': class_name,
        'super_resolution_image': img_base64,
        'image_format': 'png'
    })


if __name__ == '__main__':
    app.run()
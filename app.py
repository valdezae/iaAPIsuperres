from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
import base64
from PIL import Image
from call_model import superscale_image
from classify_image import classify_image_from_pil

import torch
import os

app = Flask(__name__)

# Path to your ResNet model weights
RESNET_WEIGHTS_PATH = 'resnet/normalsizemias.pth'  # <-- update as needed


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

    try:
        # Set device for model
        device = torch.device('cpu')
        
        # Classify the image directly using the PIL image object
        predicted_class, confidence = classify_image_from_pil(
            model_path=RESNET_WEIGHTS_PATH,
            pil_image=image,
            device=device
        )
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

    # Convert the super resolution image to base64 for JSON response
    img_buffer = io.BytesIO()
    sr_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return jsonify({
        'predicted_class_name': predicted_class,
        'confidence': float(confidence),
        'super_resolution_image': img_base64,
        'image_format': 'png'
    })


if __name__ == '__main__':
    app.run()

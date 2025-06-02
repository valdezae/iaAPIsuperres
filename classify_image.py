import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import json

# Configuration (should match the training configuration)
# These values are derived from the provided training script
CONFIG = {
    'image_size': 1024,
    'num_classes': 7,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CLASS_NAMES = ['ARCH', 'ASYM', 'CALC', 'CIRC', 'MISC', 'NORM', 'SPIC']


def create_resnet18_model(num_classes, pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def classify_image_from_pil(model_path, pil_image, device):
    """
    Classify an image directly from a PIL Image object without saving to disk.
    
    Args:
        model_path (str): Path to the model file
        pil_image (PIL.Image): PIL Image object to classify
        device (torch.device): Device to use for inference
        
    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Ensure image is in RGB mode
    img = pil_image.convert('RGB')
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = create_resnet18_model(num_classes=CONFIG['num_classes'], pretrained=True)
    
    # Load the state dictionary. Map location to CPU if CUDA is not available or not desired for inference.
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("CUDA not available. Loading model to CPU.")
        state_dict = torch.load(model_path, map_location='cpu')
        current_device = torch.device('cpu')
    else:
        state_dict = torch.load(model_path, map_location=device)
        current_device = device

    model.load_state_dict(state_dict)
    model.to(current_device)
    model.eval()
    print("Model loaded successfully.")

    # Process the image
    transform = get_transform(CONFIG['image_size'], CONFIG['mean'], CONFIG['std'])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(current_device)

    # Perform inference
    print("Classifying image...")
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_in, predicted_idx = torch.max(probabilities, 1)

    predicted_class_name = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence_in.item()

    # Free up memory
    del model
    del img_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predicted_class_name, confidence_score


# def classify_image(model_path, image_path, device):
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file not found: {model_path}")
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image file not found: {image_path}")
#
#     # Load the model
#     print(f"Loading model from {model_path}...")
#     model = create_resnet18_model(num_classes=CONFIG['num_classes'],
#                                   pretrained=True)  # Using pretrained=True as in training script.
#     # If your .pth file contains the full model
#     # including base ResNet weights, pretrained=False might also work.
#
#     # Load the state dictionary. Map location to CPU if CUDA is not available or not desired for inference.
#     if not torch.cuda.is_available() and device.type == 'cuda':
#         print("CUDA not available. Loading model to CPU.")
#         state_dict = torch.load(model_path, map_location='cpu')
#         current_device = torch.device('cpu')
#     else:
#         state_dict = torch.load(model_path, map_location=device)
#         current_device = device
#
#     model.load_state_dict(state_dict)
#     model.to(current_device)
#     model.eval()
#     print("Model loaded successfully.")
#
#     # Load and preprocess the image
#     print(f"Loading and preprocessing image {image_path}...")
#     try:
#         # Handle PGM images specifically if needed, otherwise Pillow handles many formats.
#         # The training script converts grayscale to RGB explicitly.
#         img = Image.open(image_path).convert('RGB')
#     except Exception as e:
#         raise IOError(f"Could not open or read image file: {image_path}. Error: {e}")
#
#     transform = get_transform(CONFIG['image_size'], CONFIG['mean'], CONFIG['std'])
#     img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
#     img_tensor = img_tensor.to(current_device)
#
#     # Perform inference
#     print("Classifying image...")
#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probabilities = torch.softmax(outputs, dim=1)
#         confidence_in, predicted_idx = torch.max(probabilities, 1)
#
#     predicted_class_name = CLASS_NAMES[predicted_idx.item()]
#     confidence_score = confidence_in.item()
#
#     return predicted_class_name, confidence_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single image using a pre-trained ResNet18 model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to classify.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")

    args = parser.parse_args()

    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
        if args.cuda and not torch.cuda.is_available():
            print("CUDA specified but not available, defaulting to CPU.")

    try:
        predicted_class, confidence = classify_image(args.model_path, args.image_path, device)
        print("\n--- Classification Result ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"An error occurred: {e}")

import torch
from torchvision import transforms
from PIL import Image
import sys
import os
from resnet import ResNet

def load_model(weights_path, num_classes=7, device='cpu'):
    """
    Load a trained ResNet50 model from disk
    """
    model = ResNet.ResNet50(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path):
    """
    Load and preprocess an image for inference
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet stats
    ])
    
    with Image.open(image_path) as img:
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension
    
    return img

def predict(model, image_tensor, device='cpu'):
    """
    Make a prediction using the model
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['NORM', 'CALC', 'CIRC', 'SPIC', 'MISC', 'ARCH', 'ASYM']
    class_idx = predicted.item()
    return class_idx, class_names[class_idx]

def main():
    if len(sys.argv) != 3:
        print("Usage: python use_model.py <image_path> <model_weights_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    weights_path = sys.argv[2]
    
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load the model
        print("Loading model...")
        model = load_model(weights_path, device=device)
        
        # Preprocess the image
        print(f"Processing image: {image_path}")
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        class_idx, class_name = predict(model, image_tensor, device)
        
        print("\nPrediction Results:")
        print(f"Predicted class: {class_name} (index: {class_idx})")
        
        # Class descriptions
        class_descriptions = {
            'NORM': 'Normal',
            'CALC': 'Calcification',
            'CIRC': 'Well-defined/circumscribed masses',
            'SPIC': 'Spiculated masses',
            'MISC': 'Other, ill-defined masses',
            'ARCH': 'Architectural distortion',
            'ASYM': 'Asymmetry'
        }
        
        print(f"Description: {class_descriptions.get(class_name, 'Unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

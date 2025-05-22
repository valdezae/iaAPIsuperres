import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda')

model = RealESRGAN(device, scale=2)
model.load_weights('weights/RealESRGAN_x2.pth', download=True)

def superscale_image(image: Image.Image) -> Image.Image:
    """
    Takes a PIL Image and returns the superscaled image.
    """
    return model.predict(image)

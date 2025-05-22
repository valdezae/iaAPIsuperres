import os
from PIL import Image
from call_model import superscale_image

def is_image_file(filename):
    ext = filename.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'pgm']

def batch_superscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(input_dir):
        if not is_image_file(fname):
            continue
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        try:
            img = Image.open(input_path).convert('RGB')
            sr_img = superscale_image(img)
            sr_img.save(output_path)
            print(f"Processed: {fname}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    input_dir = os.path.join("C:\\Users\\erick\\Downloads\\all-mias")
    output_dir = os.path.join("E:\\PyCharm\\resnet50\\minimias_images")
    print(f"Processing images from {input_dir} to {output_dir}")
    batch_superscale(input_dir, output_dir)

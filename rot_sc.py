import os

from PIL import Image


def rotate_images_in_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    rotated = img.rotate(90, expand=True)
                    rotated.save(output_path)
                    print(f"Rotated: {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Example usage
rotate_images_in_directory("./data/raw/1_pet_bottle", "rotated_images_pet")

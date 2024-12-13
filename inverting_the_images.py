import os
from PIL import Image, ImageOps

# Define the path to the folder containing the images
input_folder = './main_dataset/Tamil'
output_folder = './main_dataset/Sanskrit_inverted'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filter image files
    if filename.endswith(('.jpg')):  # Filter image files
        # Open each image file
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Invert the image colors
        inverted_image = ImageOps.invert(image)

        # Save the inverted image to the output folder
        output_path = os.path.join(output_folder, f'inverted_{filename}')
        inverted_image.save(output_path)

        print(f'{filename} has been inverted and saved as {output_path}')

print("All images have been inverted.")

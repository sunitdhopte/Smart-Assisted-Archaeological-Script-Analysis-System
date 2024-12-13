import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to show image using matplotlib
def show_image(title, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Step 1: Preprocessing the image (grayscale, blur, and binarize)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # filter1 = cv2.medianBlur(image,3)
    filter1 = cv2.GaussianBlur(image,(5,5),0)
    # filter2 = cv2.adaptiveThreshold(filter1, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    _, binary_image = cv2.threshold(filter1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN,kernel)
    show_image("Preprocessed Image (Binary)", binary_image)
    return binary_image

# Step 2: Find contours and filter them
def find_characters(binary_image):
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    characters = []
    min_contour_area = 50  # Minimum area to consider a contour as character (adjust as needed)
    min_aspect_ratio = 0.2  # Minimum aspect ratio (width/height) for a contour to be considered a character
    max_aspect_ratio = 5.0  # Maximum aspect ratio (to filter out very wide contours)

    # Filter contours based on area and aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            # Filter out very wide or narrow objects
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                characters.append((x, y, w, h))

    return characters

# Step 3: Draw bounding boxes around characters and save each character as an image
def draw_and_save_characters(image_path, characters, output_folder="./main_dataset/Sanskrit_Bounded1"):
    # Load original image
    image = cv2.imread(image_path)
    
    for i, (x, y, w, h) in enumerate(characters):
        # Draw rectangle around character
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 0)
        
        # Extract individual character
        character_image = image[y:y+h, x:x+w]
        # Save the character image
        cv2.imwrite(f"{output_folder}/character_{i+1300}.png", character_image)

    # Show image with bounding boxes
    show_image("Characters Detected", image)
    return image

# Main function to find and extract characters
def main(image_path):
    # Step 1: Preprocess the image
    binary_image = preprocess_image(image_path)
    
    # Step 2: Find contours and filter to detect characters
    characters = find_characters(binary_image)
    
    if characters:
        print(f"Found {len(characters)} characters.")
        # Step 3: Draw bounding boxes and save individual character images
        draw_and_save_characters(image_path, characters)
    else:
        print("No characters found.")

# Example usage:
image_path = "./datasets/Sankskrit/ancient-sanskrit-text-etched-into-260nw-243358807.jpg"
main(image_path)

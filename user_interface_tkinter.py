import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from collections import Counter
from keras.preprocessing.image import img_to_array
from skimage.feature import hog
import joblib

# Load the pre-trained models
svm_model = joblib.load('svm_character_recognition_model.pkl')
cnn_model = load_model('cnn_character_recognition_model.h5')

# Load the class names (languages)
class_names = ['Indus', 'Sanskrit', 'Tamil']  # Replace with your actual class names

# Preprocessing Function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return binary_image

# Character Segmentation
def segment_characters(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    min_contour_area = 100
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 4.0:
                characters.append((x, y, w, h))
    return characters

# Helper Function to Add Padding to Bounding Boxes
def pad_bounding_box(image, x, y, w, h, padding=5):
    h_img, w_img = image.shape
    x_pad = max(x - padding, 0)
    y_pad = max(y - padding, 0)
    w_pad = min(w + padding * 2, w_img - x_pad)
    h_pad = min(h + padding * 2, h_img - y_pad)
    return image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

# SVM Feature Extraction (HOG)
def extract_features_svm(character_image):
    resized_image = cv2.resize(character_image, (28, 28))
    features, _ = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=False)
    return features

# CNN Feature Extraction
def extract_features_cnn(character_image):
    resized_image = cv2.resize(character_image, (28, 28))
    resized_image = resized_image.reshape(28, 28, 1) / 255.0
    return resized_image

# SVM Classification
def classify_character_svm(feature_vector):
    prediction = svm_model.predict([feature_vector])
    class_index = prediction[0]
    if class_index < len(class_names):
        return class_names[class_index]
    else:
        return "Unknown"

# CNN Classification
def classify_character_cnn(feature_vector):
    prediction = cnn_model.predict(np.array([feature_vector]))
    class_index = np.argmax(prediction[0])
    if class_index < len(class_names):
        return class_names[class_index]
    else:
        return "Unknown"

# Predict characters in the whole image
def predict_characters(image_path):
    binary_image = preprocess_image(image_path)
    characters = segment_characters(binary_image)
    
    svm_predictions = []
    cnn_predictions = []

    for char in characters:
        x, y, w, h = char
        character_image = pad_bounding_box(binary_image, x, y, w, h)
        
        # SVM Prediction
        feature_vector_svm = extract_features_svm(character_image)
        svm_class = classify_character_svm(feature_vector_svm)
        svm_predictions.append(svm_class)
        
        # CNN Prediction
        feature_vector_cnn = extract_features_cnn(character_image)
        cnn_class = classify_character_cnn(feature_vector_cnn)
        cnn_predictions.append(cnn_class)
    
    # Majority prediction for SVM
    if svm_predictions:
        most_common_svm = Counter(svm_predictions).most_common(1)[0][0]
    else:
        most_common_svm = "Unknown"
    
    # Majority prediction for CNN
    if cnn_predictions:
        most_common_cnn = Counter(cnn_predictions).most_common(1)[0][0]
    else:
        most_common_cnn = "Unknown"
    
    return most_common_svm, most_common_cnn

# Tkinter UI
def upload_and_predict():
    # Clear previous image and result
    result_label.config(text="")
    image_label.config(image='')
    
    # Open file dialog to upload an image
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Display uploaded image
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        
        # Predict the characters
        svm_results, cnn_results = predict_characters(file_path)
        
        # Display the final result
        result_text = f"SVM Prediction: {svm_results}\nCNN Prediction: {cnn_results}"
        result_label.config(text=result_text)

# Create the Tkinter window
root = tk.Tk()
root.title("Character Recognition System")

# Button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_button.pack()

# Label to display the uploaded image
image_label = tk.Label(root)
image_label.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack()

# Start Tkinter main loop
root.mainloop()

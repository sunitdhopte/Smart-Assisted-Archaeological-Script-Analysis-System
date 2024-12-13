from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import cv2
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from werkzeug.utils import secure_filename

app = Flask(__name__)

svm_model = joblib.load('svm_character_recognition_model.pkl')
cnn_model = load_model('cnn_character_recognition_model.h5')

app.config['UPLOAD_FOLDER'] = './uploads'

language_mapping = {
    0: 'Indus',
    1: 'Sanskrit',
    2: 'Tamil',
}

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filter1 = cv2.medianBlur(image,3)
    _, binary_image = cv2.threshold(filter1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN,kernel)
    return binary_image

def segment_characters(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    min_contour_area = 50
    min_aspect_ratio = 0.5
    max_aspect_ratio = 5.0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                characters.append((x, y, w, h))
    return characters

def classify_image(image_path):
    binary_image = preprocess_image(image_path)
    characters = segment_characters(binary_image)

    true_labels = []
    svm_predictions = []
    cnn_predictions = []

    for char in characters:
        x, y, w, h = char
        character_image = binary_image[y:y+h, x:x+w]
        
        true_label = 0 
        true_labels.append(true_label)  

        # SVM Prediction
        feature_vector_svm = cv2.resize(character_image, (28, 28)).flatten()
        svm_pred = svm_model.predict([feature_vector_svm])
        svm_predictions.append(svm_pred[0])
        
        # CNN Prediction
        feature_vector_cnn = cv2.resize(character_image, (28, 28)).reshape(28, 28, 1) / 255.0
        cnn_pred = cnn_model.predict(np.array([feature_vector_cnn]))
        cnn_predictions.append(np.argmax(cnn_pred[0]))

    svm_majority = max(set(svm_predictions), key=svm_predictions.count)
    cnn_majority = max(set(cnn_predictions), key=cnn_predictions.count)
    
    svm_language_name = language_mapping[svm_majority]
    cnn_language_name = language_mapping[cnn_majority]

    return svm_language_name, cnn_language_name, true_labels, svm_predictions, cnn_predictions

def plot_confusion_matrix(y_true, y_pred, model_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'./static/{model_name}_confusion_matrix.png')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            svm_result, cnn_result, true_labels, svm_predictions, cnn_predictions = classify_image(file_path)

            plot_confusion_matrix(true_labels, svm_predictions, 'SVM')
            plot_confusion_matrix(true_labels, cnn_predictions, 'CNN')
            
            return render_template('index.html', svm_language=svm_result, cnn_language=cnn_result, image_filename=filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
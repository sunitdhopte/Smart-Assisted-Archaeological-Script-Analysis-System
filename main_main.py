import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import joblib
from collections import Counter
import time
import seaborn as sns

def load_dataset(data_dir):
    images = []
    labels = []
    languages = os.listdir(data_dir)
    
    for language in languages:
        language_dir = os.path.join(data_dir, language)
        if not os.path.isdir(language_dir):
            continue
        
        for image_name in os.listdir(language_dir):
            image_path = os.path.join(language_dir, image_name)
            image = load_img(image_path, target_size=(28, 28), color_mode='grayscale')
            image = img_to_array(image)
            images.append(image)
            labels.append(language)
    
    images = np.array(images)
    labels = np.array(labels)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return images, labels, label_encoder.classes_

data_dir = './main_dataset'
images, labels, class_names = load_dataset(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)
joblib.dump(svm_model, './models/svm_character_recognition_model.pkl')

X_train_cnn = X_train / 255.0
X_test_cnn = X_test / 255.0

y_train_cnn = to_categorical(y_train, num_classes=len(class_names))
y_test_cnn = to_categorical(y_test, num_classes=len(class_names))

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test_cnn))
cnn_model.save('./models/cnn_character_recognition_model.h5')

def show_image(title, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filter1 = cv2.GaussianBlur(image,(5,5),0)
    _, binary_image = cv2.threshold(filter1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN,kernel)
    show_image("Preprocessed Image (Binary)", binary_image)
    return binary_image

def segment_characters(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    min_contour_area = 50  
    min_aspect_ratio = 0.2  
    max_aspect_ratio = 5.0  
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                characters.append((x, y, w, h))
    
    # rectangles around segmented characters for visualization
    segmented_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in characters:
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    show_image("Character Segmentation", segmented_image)
    return characters

def extract_features_svm(character_image):
    resized_image = cv2.resize(character_image, (28, 28)).flatten()
    return resized_image

def extract_features_cnn(character_image):
    resized_image = cv2.resize(character_image, (28, 28))
    resized_image = resized_image.reshape(28, 28, 1) / 255.0 
    return resized_image

def classify_character_svm(feature_vector):
    prediction = svm_model.predict([feature_vector])
    return prediction[0]

def classify_character_cnn(feature_vector):
    prediction = cnn_model.predict(np.array([feature_vector]))
    return np.argmax(prediction[0])

def evaluate_model(y_true, y_pred, model_name):
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
def measure_prediction_speed(model_func, feature_vector):
    start_time = time.time()
    model_func(feature_vector)
    return time.time() - start_time

def main(image_path):
    binary_image = preprocess_image(image_path)
    characters = segment_characters(binary_image)
    
    svm_predictions = []
    cnn_predictions = []
    svm_times = []
    cnn_times = []
    
    for char in characters:
        x, y, w, h = char
        character_image = binary_image[y:y+h, x:x+w]
        
        feature_vector_svm = extract_features_svm(character_image)
        svm_time = measure_prediction_speed(classify_character_svm, feature_vector_svm)
        svm_times.append(svm_time)
        class_index_svm = classify_character_svm(feature_vector_svm)
        svm_predictions.append(class_index_svm)
        
        feature_vector_cnn = extract_features_cnn(character_image)
        cnn_time = measure_prediction_speed(classify_character_cnn, feature_vector_cnn)
        cnn_times.append(cnn_time)
        class_index_cnn = classify_character_cnn(feature_vector_cnn)
        cnn_predictions.append(class_index_cnn)
    
    if svm_predictions:
        most_common_svm = Counter(svm_predictions).most_common(1)[0][0]
        print(f"The majority of characters belong to language (SVM): {class_names[most_common_svm]}")
    else:
        print("No characters were detected in the image (SVM).")
    
    if cnn_predictions:
        most_common_cnn = Counter(cnn_predictions).most_common(1)[0][0]
        print(f"The majority of characters belong to language (CNN): {class_names[most_common_cnn]}")
    else:
        print("No characters were detected in the image (CNN).")

    print(f"Average SVM prediction time per character: {np.mean(svm_times):.5f} seconds")
    print(f"Average CNN prediction time per character: {np.mean(cnn_times):.5f} seconds")
    
    evaluate_model(y_test, svm_model.predict(X_test_flat), "SVM")
    evaluate_model(np.argmax(cnn_model.predict(X_test_cnn), axis=1), y_test, "CNN")
    

image_path = "./datasets/Sankskrit/Sanskrit_Inscription_at_Qutub_Complex.jpeg.jpg"
main(image_path)
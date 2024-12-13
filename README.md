# Smart Assisted Archaeological Script Analysis System

## Project Overview

*Smart Assisted Archaeological Script Analysis System** is a deep learning and machine learning-based tool designed to analyze and detect the language of ancient scripts. The system uses bounding boxes to identify text regions in images, classifies the detected text into different languages, and evaluates performance using models like CNN (Convolutional Neural Networks) and SVM (Support Vector Machines).

---

## Features

- **Language Detection**: Automatically detects the language of the text from images.
- **Bounding Box Detection**: Identifies regions of interest in the image where text is located.
- **Model Comparison**: Compares the performance of CNN (deep learning) and SVM (machine learning).
- **Evaluation Metrics**: Utilizes confusion matrices for performance evaluation.
- **Web Interface**: A user-friendly interface for uploading images and viewing results.

---

## Technologies Used

- **Programming Language**: Python, JavaScript
- **Frameworks**: TensorFlow/Keras, OpenCV
- **Machine Learning**: Support Vector Machines (SVM)
- **Deep Learning**: Convolutional Neural Networks (CNN)
- **Web Development**: HTML5, CSS3 (Frontend), Flask (Backend)

---

## Project Workflow

1. **Input**: User uploads an image containing ancient scripts.
2. **Preprocessing**: 
   - Image processing using OpenCV to detect text regions.
   - Bounding boxes generated around text.
3. **Feature Extraction**:
   - Text features extracted for analysis.
4. **Classification**:
   - CNN for deep learning-based classification.
   - SVM for traditional machine learning classification.
5. **Output**:
   - Language detected.
   - Evaluation report displayed (confusion matrix and model comparison).

---
---

## Getting Started

### Prerequisites

1. Python 3.8+
2. Virtual Environment (optional but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SunitDhopte/Archaeological_Script_Analysis.git
   cd Archaeological_Script_Analysis
   ```
   
2. Run the application:
   ```bash
   python app.py
   ```

3. Access the application at `http://localhost:5000`.

---

## Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| CNN   | 95%      | 94%       | 96%    |
| SVM   | 85%      | 82%       | 88%    |

---

## Future Enhancements

- Adding support for additional ancient languages.
- Improving bounding box detection with YOLO or similar frameworks.
- Incorporating cross-validation for robust model evaluation.

---

## Contact

**Sunit Dhopte**  
For queries, reach out at [sunitdhopte22@gmail.com](mailto:sunitdhopte22@gmail.com).

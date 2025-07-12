# 🧠 Handwritten Character Recognition using CNN (OCR Project)

This project is built as part of the CodeAlpha Machine Learning Internship - **Task 3: Handwritten Character Recognition**. It uses a **Convolutional Neural Network (CNN)** to identify handwritten digits from the **MNIST dataset**, achieving high accuracy with a clean and minimal deep learning architecture.

---

## ✅ Objective
To develop an OCR (Optical Character Recognition) system that can accurately classify **handwritten digits (0-9)** using image processing and deep learning techniques.

---

## 🔍 Key Features

- 🔢 **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) (60,000 training + 10,000 testing images)
- 🧠 **Model**: CNN (Convolutional Neural Network) using TensorFlow/Keras
- 📈 **Accuracy**: ~98% on test data
- 📦 **Output**: A saved `.h5` model file and prediction visualizations
- 🖼️ **Prediction**: Test and visualize model predictions on random samples

---

## 🛠️ Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy, Matplotlib
- MNIST Dataset

---

## 🚀 How It Works

1. **Load and Preprocess Data**  
   Normalize and reshape the MNIST grayscale images.

2. **Train CNN Model**  
   Build and train a 7-layer CNN to classify the images into 10 digit classes.

3. **Evaluate Model**  
   Measure accuracy and visualize predictions and performance.

4. **Save and Reuse**  
   Save the trained model (`mnist_cnn_model.h5`) for future use or deployment.

---

## 📷 Sample Output

| Input Image | Predicted Digit |
|-------------|------------------|
| ![sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) | ✅ Recognized with ~98% accuracy |

---

## 💾 Installation

```bash
git clone https://github.com/yourusername/handwritten-digit-recognition
cd handwritten-digit-recognition
pip install -r requirements.txt
python task3_handwritten_digit_recognition.py

# Digital-Image-Processing
**Project Title:**
Digit Recognition & Classification System using Logistic Regression and CNN

**📌 Objective:**
To build a Python-based system that recognizes handwritten digits from an image and classifies the detected number into categories like CNIC, Mobile, Passport, ATM, or Check Book based on digit count.

**📊 Technologies & Libraries Used:**
Python

OpenCV – Image preprocessing and digit segmentation

PyTorch – Convolutional Neural Network (CNN)

Scikit-learn – Logistic Regression

Matplotlib – Image visualization

**🧠 Machine Learning Models:**
Logistic Regression

Trained using mnist_784 dataset from OpenML.

Multi-class classification with max_iter=100.

Convolutional Neural Network (CNN)

1 Conv2D layer + MaxPooling → Fully connected layers.

Trained on MNIST dataset for 3 epochs.

**🔍 Features:**
Input: User-provided image path.

Digit extraction using thresholding and contour detection.

Model choice: Logistic Regression or CNN.

Auto-classifies recognized number as:

9 digits → Passport

11 digits → Mobile

13 digits → CNIC

16 digits → ATM

18 digits → Check Book

**📥 User Input:**
Image path (e.g., digits.png)

Model selection: 1 (Logistic) or 2 (CNN)

Expected type (Passport, CNIC, etc.)

**✅ Output:**
Recognized digit sequence (e.g., 3520123456789)

Auto-classification (e.g., CNIC)

Match/mismatch with user-selected type

Displays image for reference

**📁 How to Run:**
python digit_recognition.py
Ensure the digit image is clear, in grayscale, and digits are separated by white space.

**📌 Note:**
CNN model is saved as cnn_digit_model.pth.

Can be enhanced with GUI, image cropping, or digit alignment.

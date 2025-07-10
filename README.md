# 🐠 Sea Animals Image Classification using Handcrafted Features

## 📌 Overview

This project focuses on classifying images of various **sea animals** into multiple categories using **traditional machine learning techniques** (SVM, Random Forest), without deep learning or transfer learning. The classification is performed by extracting **handcrafted features** from each image — specifically, **Color Histograms**, **Local Binary Patterns (LBP)**, and **Hu Moments**.

## 🧠 Project Goals

- Perform **image classification** without neural networks
- Use classical ML techniques and handcrafted features
- Train models on a multi-class dataset
- Evaluate models using proper metrics
- Save the final model for reuse

## 🗂️ Dataset

- **Source**: [Sea Animals Image Dataset - Kaggle]
- **Format**: ZIP archive, containing class-wise folders
- **Classes**: crab, fish, lobster, octopus, seahorse, shark, starfish, etc.
- **Size**: Large, with thousands of images

## ⚙️ Technologies Used

- Python (Google Colab)
- OpenCV (cv2)
- Scikit-learn
- Scikit-image
- NumPy, Matplotlib, Seaborn
- joblib (model saving)

## 🔁 Workflow Summary

1. **Google Drive Integration**  
   Mount Drive and extract ZIP.

2. **Image Preprocessing**  
   - Resize images to 128x128  
   - Convert to grayscale / HSV

3. **Feature Extraction**  
   - **Color Histogram** (HSV space)  
   - **Local Binary Patterns (LBP)**  
   - **Hu Moments** (shape descriptors)  
    All combined into a feature vector

4. **Data Preparation**  
   - Label encoding  
   - Train-test split (80:20)

5. **Model Training**  
   - SVM (RBF kernel)  
   - Random Forest (n_estimators = 50)  
   - Gradient Boosting (skipped for compute efficiency)

6. **Model Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - Confusion matrix (visualized)

7. **Model Saving**  
   - joblib.dump(models["Random Forest"], "sea_animals_rf_model.pkl")

**Key Learnings**

1. Handcrafted features are still powerful for image classification.
2. Combining texture, color, and shape improves accuracy.
3. SVM and Random Forest offer fast, reliable classification.
4. Gradient Boosting was avoided due to limited resources in Colab.

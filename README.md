# Melanoma Detection using Custom CNN

## Table of Contents
* [General Information](#general-information)
* [Project Pipeline](#project-pipeline)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [Model Building](#model-building)
* [Handling Class Imbalances](#handling-class-imbalances)
* [Findings & Conclusions](#findings--conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
Melanoma is a severe type of skin cancer that accounts for 75% of skin cancer deaths. This project aims to develop a CNN-based deep learning model that can accurately detect melanoma from skin images. A reliable AI solution can help dermatologists diagnose melanoma early, reducing manual effort and saving lives.

## Project Pipeline
1. **Data Understanding & Preparation**:
   - Load and explore the dataset.
   - Define paths for training and testing images.
   
2. **Dataset Creation**:
   - Create train and validation datasets.
   - Resize images to 180x180 pixels.
   - Set batch size to 32.

3. **Dataset Visualization**:
   - Display one image from each of the nine classes.

4. **Model Building & Training (Initial Model)**:
   - Build a custom CNN model from scratch.
   - Rescale images to normalize pixel values between (0,1).
   - Choose an appropriate optimizer and loss function.
   - Train the model for ~20 epochs.
   - Analyze overfitting/underfitting issues.

5. **Applying Data Augmentation**:
   - Implement augmentation strategies to resolve overfitting/underfitting.
   - Retrain the model for ~20 epochs and compare results.

6. **Handling Class Imbalances**:
   - Analyze class distribution to identify imbalances.
   - Use the `Augmentor` library to balance the dataset.

7. **Model Training on Balanced Data**:
   - Train the model for ~30 epochs on the rectified dataset.
   - Compare performance and check if imbalances were handled effectively.

## Technologies Used
- **Python** - 3.8+
- **TensorFlow/Keras** - Deep Learning Framework
- **OpenCV** - Image Processing
- **Matplotlib & Seaborn** - Data Visualization
- **Augmentor** - Data Augmentation
- **Google Colab** - GPU Training

## Dataset
The dataset consists of 2357 images categorized into nine oncological diseases:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

The dataset is sourced from the **International Skin Imaging Collaboration (ISIC)**.

## Model Building
- **CNN Architecture**:
  - Conv2D layers with increasing filters.
  - Batch Normalization and Dropout for regularization.
  - Flattening and Dense layers for classification.
  - Softmax activation for multiclass classification.
- **Optimizer & Loss Function**:
  - Adam optimizer
  - Sparse Categorical Cross-Entropy loss

## Handling Class Imbalances
- **Class Distribution Analysis**:
  - Identify underrepresented classes.
  - Use `Augmentor` to generate synthetic images.
- **Training on Balanced Data**:
  - Model retrained on the rectified dataset.
  - Performance compared with previous models.

## Findings & Conclusions
- The initial model faced slight overfitting.
- Data augmentation improved generalization.
- Handling class imbalances helped boost classification accuracy.
- The final model demonstrated improved robustness and accuracy in melanoma detection.

## Acknowledgements
- Dataset provided by **ISIC**.
- Inspired by dermatology AI applications.
- Research references on CNN-based skin cancer detection.

## Contact
Created by **[@yourgithubusername]** - Feel free to reach out for collaboration or questions!


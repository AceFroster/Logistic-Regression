# Breast Cancer Prediction with Logistic Regression
Welcome to my Breast Cancer Prediction project! 🎯 In this project, I used Logistic Regression to predict whether a breast tumor is Malignant or Benign based on a dataset from Kaggle. The goal was to develop a machine learning model that can help diagnose breast cancer with high accuracy.

## Table of Contents
 Project Overview

 Dataset
 
 What I Did:
   1. Data Preprocessing
   2. Feature Scaling
   3. Train/Test Split
   4. Model Building
   5. Evaluation

 Technologies Used

## Project Overview
In this project, I used Logistic Regression, a simple but powerful machine learning algorithm, to classify breast cancer tumors as either Malignant or Benign. I worked with a dataset from Kaggle, which contains various measurements about cell nuclei present in breast cancer biopsies. The goal was to preprocess the data, train a model, and evaluate its performance.

By the end of the project, the Logistic Regression model achieved an impressive 98% accuracy on the test set, which shows the potential of machine learning in healthcare applications.

## Dataset
The dataset used in this project is located in the Dataset folder within this repository. It contains various features extracted from breast cancer biopsies, such as texture, area, smoothness, and more. Each sample is labeled as either Malignant (M) or Benign (B).

The dataset is already pre-processed in the sense that the features are numeric, but additional cleaning and transformations were done during the project to make the data ready for modeling.

## What I Did
Here’s a breakdown of the steps I took in this project:

1. Data Preprocessing
    Data cleaning and preparation are some of the most important steps in any machine learning project. For this project, I:

    Removed unnecessary columns such as Unnamed: 32 (an empty column) and id (which didn’t contribute to the diagnosis prediction).
   
    Encoded the target variable diagnosis to turn the categorical labels into binary values. Specifically, I converted:
   
      M (Malignant) → 1
   
      B (Benign) → 0
   
    This is important for machine learning models to understand the data properly.

2. Feature Scaling
   
   Since the dataset contained features with varying scales (e.g., radius, area, smoothness), I needed to normalize the data to ensure that the model treats each feature equally. To do this, I used 
   StandardScaler to scale the data and bring all features into the same range.

   Feature scaling is crucial when using algorithms like Logistic Regression because it ensures that no single feature dominates the model’s learning process due to differences in scale.

3. Train/Test Split
   
   To evaluate the model’s performance, I divided the dataset into two parts:

      70% of the data for training
   
      30% of the data for testing
   
   This allows me to train the model on one portion of the data and then test it on an unseen portion to simulate how the model would perform in real-world scenarios.

4. Model Building
   
   With the data prepared and split, I built a Logistic Regression model using the Scikit-learn library. Logistic Regression is ideal for binary classification problems like this one because it outputs 
   probabilities for each class, making it a great fit for predicting whether the tumor is malignant or benign.

5. Evaluation
   
   After training the model, I evaluated its performance using the accuracy score. The model achieved a 98% accuracy, meaning that it correctly predicted the diagnosis of 98% of the test samples. This 
   result is a clear indicator that the model is performing well!

## Technologies Used
Here are the tools and libraries I used to complete this project:

1) Python: The main programming language used for the entire project.
2) Pandas: To manipulate and preprocess the dataset.
3) NumPy: For numerical operations and handling arrays.
4) Scikit-learn: For building and training the Logistic Regression model, as well as evaluating its performance.
5) Matplotlib and Seaborn (optional): For data visualization, if any visualizations were included in the project.
 

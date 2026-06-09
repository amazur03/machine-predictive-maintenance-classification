# Predictive Maintenance Classification with Custom SVM

This repository contains a complete machine learning pipeline for predictive maintenance. The core of the project is a custom implementation of a Support Vector Machine (SVM) classifier built from scratch using NumPy, optimized with Stochastic Gradient Descent (SGD). 

The custom model is evaluated on a synthetic dataset designed to mimic real-world industrial predictive maintenance scenarios, and compared against standard industry baselines from scikit-learn.

## Key Features

* Custom SVM Implementation: Built entirely from scratch, featuring L2 regularization, Hinge Loss tracking, learning rate decay, and built-in handling for imbalanced classes via dynamically calculated class weights.
* Advanced Feature Engineering: Creation of domain-specific features (e.g., Temp_diff, Power_estimate, Torque_per_speed, Wear_squared) to capture complex physical relationships between machine sensors.
* Data Leakage Prevention: A custom Standardization class that strictly computes Z-score parameters (mean and standard deviation) only on the training set, applying them safely to the test set.
* Hyperparameter Tuning: A dedicated script utilizing Grid Search to find the optimal learning_rate and lambda_param targeting the F1-Score for the minority (failure) class.
* Comprehensive Visualizations: Includes learning curves to monitor model convergence and Seaborn-based confusion matrices for intuitive performance evaluation.

## Project Structure

* main.py - The main entry point. Orchestrates data loading, trains the custom SVM, plots the learning curve, and runs the scikit-learn baseline models for comparison.
* svm_model.py - Contains the SVM class (the from-scratch implementation using NumPy).
* data_preprocessing.py - Handles Kaggle dataset downloading, data cleaning, one-hot encoding, feature engineering, and stratified train-test splitting.
* utils.py - Contains helper tools, including the custom Standardization class and the plot_confusion visualization function.
* tune_hyperparameters.py - A standalone script for running Grid Search to optimize the custom SVM's parameters.

## Dataset

The project automatically downloads the AI4I 2020 Predictive Maintenance Dataset directly from Kaggle using kagglehub.
* Source: Machine Predictive Maintenance Classification (Kaggle)
* Type: Synthetic dataset reflecting actual industrial parameters and failure modes.
* The target variable (Target) indicates whether a machine failure occurred.

## Installation & Requirements

Ensure you have Python 3.8+ installed. First, clone the repository, then install the required dependencies:

```bash
pip install -r requirements.txt
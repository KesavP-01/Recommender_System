
## Overview

This project implements a Recommender System using deep learning with TensorFlow. It includes feature engineering and data preprocessing to improve binary classification accuracy.

The model processes user interaction datasets to predict preferences and generate recommendations.
## Key Features

**Deep Learning Model:** Implements a neural network using TensorFlow, optimized for binary classification.

**Data Processing:** Includes preprocessing steps such as feature scaling, label encoding, and interaction term generation.

**Model Evaluation:** Comprehensive evaluation of model performance using accuracy and ROC AUC metrics.
## Installation

To set up the project locally, follow these steps:


Clone this repository:
```bash
  git clone https://github.com/KesavP-01/Recommender_System.git cd Recommender_System-main
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Run the data generation script:
```bash
python scripts/genratedata.py
```
Train the model:
```bash
python models/model_train.py
```
Evaluate the model:
```bash
python models/evaluate_model.py
```
## Usage

**Training the Model:** The deep learning model is trained using **models/model_train.py**, which loads the preprocessed data and trains the neural network. The trained model is saved as **rec_model.h5** in the models/ directory.

**Evaluating the Model:** Once trained, you can evaluate the model using **models/evaluate_model.py**. This script will load the test data and calculate metrics like accuracy and ROC AUC to assess the model's performance.
## Data

The data/ folder contains the dataset used for training and evaluating the model:

**interactions.csv:** Contains the original interaction dataset.

**X_test.npy and y_test.npy:** Preprocessed test data in NumPy format for evaluating the model's performance.

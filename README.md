# NLP Comment Classification System

A machine learning pipeline for classifying user-generated comments into predefined categories using Natural Language Processing, feature engineering, feature selection, and supervised machine learning.

The project explores multiple classification approaches and combines textual, numerical, and categorical information to improve predictive performance.

---

## Project Overview

This project focuses on building an end-to-end NLP classification pipeline capable of learning from textual comments and predicting categories for unseen data.

The workflow covers:

- Exploratory Data Analysis
- Data preprocessing
- Missing-value handling
- Text preprocessing
- TF-IDF feature extraction
- Numerical feature engineering
- Categorical feature encoding
- Chi-square feature selection
- Dimensionality reduction experiments
- Multiple machine learning models
- Class imbalance handling
- Model comparison
- Validation and performance evaluation
- Final model selection
- Test-set prediction

The final selected approach combines feature-selected TF-IDF text features with numerical metadata and uses a class-weighted Logistic Regression classifier.

---

## Key Results

| Metric | Result |
|---|---:|
| Training Samples | 198,000 |
| Test Samples | 102,000 |
| Original TF-IDF Features | 20,000 |
| Selected TF-IDF Features | 5,000 |
| Feature Reduction | 75% |
| Best Validation Accuracy | 90.64% |
| Best Model | Logistic Regression |
| Feature Selection | SelectKBest + Chi-square |
| Text Representation | TF-IDF |
| Validation Strategy | Stratified Train/Validation Split |

The best-performing model was Logistic Regression using selected TF-IDF features combined with numerical features.

---

## Problem Statement

The objective of the project is to classify user comments into predefined categories using Natural Language Processing and machine learning techniques.

Given a comment and associated metadata, the system learns patterns from labeled training data and predicts the appropriate category for unseen comments.

The main goals were:

1. Understand the structure and distribution of the dataset.
2. Extract meaningful information from textual comments.
3. Incorporate useful numerical and categorical metadata.
4. Compare multiple machine learning approaches.
5. Reduce unnecessary feature dimensionality.
6. Address class imbalance.
7. Select the most effective model based on validation performance.
8. Generate predictions for the test dataset.

---

## Dataset

The project uses a comment classification dataset containing:

- `198,000` training samples
- `102,000` test samples
- Textual comment data
- Numerical metadata
- Categorical metadata
- Target labels in the training dataset

The original Kaggle environment contained:

```text
Sample.csv
train.csv
test.csv

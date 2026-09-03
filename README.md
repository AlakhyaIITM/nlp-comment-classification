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
````

The dataset itself is not included in this repository.

Please refer to the original Kaggle dataset/challenge for dataset access and licensing information.

---

## Features

The model uses multiple types of information.

### Text Features

The primary input is the user-generated comment.

Text is transformed into numerical representations using TF-IDF.

### Numerical Features

The project uses the following numerical attributes:

```text
emoticon_1
emoticon_2
emoticon_3
upvote
downvote
if_1
if_2
disability
```

These features are standardized using `StandardScaler`.

### Categorical Features

The following categorical attributes are encoded using `OneHotEncoder`:

```text
race
religion
gender
```

The encoding produced 22 categorical features.

---

## Project Workflow

```text
Raw Dataset
     |
     v
Data Loading
     |
     v
Exploratory Data Analysis
     |
     v
Missing Value Handling
     |
     +----------------------+
     |                      |
     v                      v
Text Features          Structured Features
     |                      |
     v                      |
TF-IDF                 Numerical Features
     |                      |
     |                  StandardScaler
     |                      |
     |                      v
     |                 Categorical Features
     |                      |
     |                 OneHotEncoder
     |                      |
     +----------+-----------+
                |
                v
        Feature Engineering
                |
                v
       Feature Selection
       SelectKBest + Chi2
                |
                v
        Model Training
                |
                v
       Model Comparison
                |
                v
       Class Imbalance
          Handling
                |
                v
       Model Evaluation
                |
                v
        Final Model
                |
                v
      Test Predictions
```

---

# Exploratory Data Analysis

The first stage of the project focuses on understanding the dataset before model development.

The analysis includes:

* Dataset dimensions
* Data types
* Missing values
* Numerical feature statistics
* Class distribution
* Comment characteristics
* Comment length analysis
* Feature distributions

Initial dataset inspection was performed using Pandas, including `info()`, `describe()`, missing-value analysis, and sample inspection.

---

# Data Preprocessing

Missing values were examined across the dataset.

The preprocessing strategy included handling missing values according to feature type.

For textual comments:

```text
Missing comment -> ""
```

For selected categorical attributes:

```text
Missing value -> "unknown"
```

This allowed the downstream feature extraction and encoding pipelines to operate without dropping potentially useful observations.

---

# Text Feature Engineering

## TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) was used to transform comments into numerical feature vectors.

The main configuration was:

```python
TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
```

This produced:

```text
20,000 text features
```

The use of both unigrams and bigrams allows the model to capture individual terms as well as short combinations of terms.

---

# Numerical Feature Engineering

The numerical features were standardized using `StandardScaler`.

```python
StandardScaler()
```

The scaled numerical features were then combined with the TF-IDF representation using sparse feature concatenation.

The initial combined feature representation contained:

```text
20,000 TF-IDF features
+
8 numerical features
=
20,008 features
```

Example training representation:

```text
Training   : 158,400 × 20,008
Validation : 39,600 × 20,008
```

---

# Categorical Feature Engineering

Categorical attributes were transformed using `OneHotEncoder`.

The project encoded:

```text
race
religion
gender
```

with:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
```

The resulting representation contained:

```text
22 encoded categorical features
```

This allowed structured categorical information to be incorporated alongside textual features.

---

# Feature Selection

## SelectKBest + Chi-square

High-dimensional TF-IDF representations can contain many features that contribute little to the classification task.

To reduce the feature space, the project used:

```python
SelectKBest(
    score_func=chi2,
    k=5000
)
```

The original TF-IDF representation contained:

```text
20,000 features
```

After feature selection:

```text
5,000 features
```

This resulted in:

```text
75% feature reduction
```

The selected features were determined using the chi-square statistical test.

### Feature Reduction

```text
Original TF-IDF
20,000 features
        |
        | SelectKBest + Chi-square
        v
Selected TF-IDF
5,000 features
```

The reduction was calculated as:

```text
1 - (5,000 / 20,000)
= 75%
```

---

# Dimensionality Reduction Experiment

The project also experimented with Truncated SVD to reduce the high-dimensional sparse feature space into a compact representation.

The experiment used:

```python
TruncatedSVD(
    n_components=300,
    random_state=42
)
```

This was explored as an alternative representation, but the final selected model used feature selection rather than the SVD representation.

---

# Machine Learning Models

Multiple classification approaches were evaluated during the project.

The experiments included:

### 1. Logistic Regression

A TF-IDF-based Logistic Regression model was used as a strong baseline.

### 2. Multinomial Naive Bayes

Naive Bayes was evaluated as a traditional text classification baseline.

### 3. Linear Support Vector Machine

A LinearSVC model was evaluated using text and numerical features.

### 4. Logistic Regression with Feature Fusion

Textual features were combined with numerical features before training the Logistic Regression model.

### 5. Feature Selection + Numerical Feature Fusion

The best-performing approach combined:

* Selected TF-IDF features
* Numerical features
* Feature scaling
* Class weighting
* Logistic Regression

### 6. Voting Classifier

An ensemble approach was also evaluated.

### 7. Hyperparameter-Tuned Logistic Regression

Additional Logistic Regression tuning was explored to determine whether further performance improvements could be obtained.

---

# Class Imbalance Handling

Class imbalance was considered during model development.

The final Logistic Regression model used class weights:

```python
{
    0: 1,
    1: 3,
    2: 2,
    3: 5
}
```

This assigns greater importance to selected minority classes during training.

The project found that class weighting substantially improved recall for minority classes.

---

# Model Comparison

The project evaluated different combinations of:

* Text-only features
* Text + numerical features
* Feature-selected text
* Feature-selected text + numerical features
* Ensemble methods
* Hyperparameter tuning

The results demonstrated that feature engineering and feature selection were more impactful than simply increasing model complexity.

The strongest overall approach was:

```text
TF-IDF
   |
   v
Chi-square Feature Selection
   |
   v
5,000 Selected Text Features
   |
   +---- Numerical Features
   |
   v
Feature Fusion
   |
   v
Class-Weighted Logistic Regression
   |
   v
Final Prediction
```

---

# Final Model

The final selected model was:

```text
Logistic Regression
```

with:

* TF-IDF text representation
* Chi-square feature selection
* 5,000 selected text features
* Numerical feature fusion
* Class weighting
* Maximum iterations configured for convergence

The final model achieved:

## 90.64% Validation Accuracy

This was the best validation performance among the evaluated approaches.

---

# Key Findings

Several observations emerged from the experiments.

### TF-IDF provides a strong baseline

TF-IDF effectively captured useful textual patterns and provided a strong starting point for the classification task.

### Feature engineering improved performance

Combining textual information with numerical metadata produced a meaningful improvement compared with using text features alone.

### Feature selection reduced noise

Reducing the TF-IDF representation from 20,000 to 5,000 features helped remove less useful features while maintaining strong classification performance.

### Class weighting improved minority-class recall

Applying class weights helped the model pay greater attention to underrepresented classes.

### Hyperparameter tuning had limited additional impact

Additional parameter tuning produced only marginal gains compared with the improvements obtained through feature engineering.

### Ensemble methods did not outperform the best model

The evaluated ensemble approach did not exceed the performance of the feature-selected Logistic Regression model.

---

# Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn

## NLP

* TF-IDF
* N-grams
* Text preprocessing
* Feature selection
* Chi-square statistical testing

## Feature Engineering

* StandardScaler
* OneHotEncoder
* Sparse feature fusion
* SelectKBest

## Dimensionality Reduction

* TruncatedSVD

## Model

* Logistic Regression
* Multinomial Naive Bayes
* LinearSVC
* Voting Classifier

## Development Environment

* Jupyter Notebook
* Kaggle

---

# Repository Structure

```text
nlp-comment-classification/
│
├── README.md
├── nlp_comment_classification.ipynb
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   └── README.md
│
└── results/
    └── model_results.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/nlp-comment-classification.git
```

Move into the project directory:

```bash
cd nlp-comment-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The main project is provided as a Jupyter Notebook.

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
nlp_comment_classification.ipynb
```

Before running the notebook, make sure the dataset files are available locally and update the dataset path used by the notebook.

---

# Reproducibility

The project uses a stratified train-validation split with:

```python
random_state=42
```

This helps maintain consistent class proportions between the training and validation sets and provides reproducible experiments.

---

# Results Summary

The most important optimization stages were:

```text
20,000 TF-IDF Features
        |
        v
Chi-square Feature Selection
        |
        v
5,000 Selected Features
        |
        | 75% reduction
        v
Numerical Feature Fusion
        |
        v
Class-Weighted Logistic Regression
        |
        v
90.64% Validation Accuracy
```

---

# What This Project Demonstrates

This project demonstrates practical experience with:

* NLP classification
* Large-scale text feature extraction
* TF-IDF
* Feature engineering
* Feature selection
* Numerical and categorical feature integration
* Sparse feature representations
* Class imbalance
* Model comparison
* Logistic Regression
* Support Vector Machines
* Naive Bayes
* Ensemble learning
* Dimensionality reduction
* Validation and evaluation
* End-to-end machine learning workflows

---

# Future Improvements

Potential future improvements include:

* Experimenting with transformer-based language models
* Testing additional text preprocessing techniques
* Exploring word and character-level representations
* Performing systematic hyperparameter optimization
* Evaluating additional metrics across all classes
* Investigating more advanced class imbalance techniques
* Building a lightweight inference API
* Creating an interactive prediction interface
* Packaging the final model into a deployable application

These improvements are not part of the current implementation.

---

# Kaggle

The original notebook and project work were developed on Kaggle.

**Kaggle Notebook:**
[https://www.kaggle.com/code/alakhyasarkar/23f1001050-notebook-t12026](https://www.kaggle.com/code/alakhyasarkar/23f1001050-notebook-t12026)

---

# Author

**Alakhya Sarkar**

B.S. in Data Science and Applications
Indian Institute of Technology Madras


---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

````

### One important change I would make before you upload it

I **would not blindly keep the `requirements.txt` and `results/model_results.md` structure unless we actually create those files**. The README should accurately reflect the repository.

For your first GitHub version, I recommend this simpler structure:

```text
nlp-comment-classification/
│
├── README.md
├── nlp_comment_classification.ipynb
├── requirements.txt
├── LICENSE
├── .gitignore
└── data/
    └── README.md
````

# NLP Comment Classification System

An end-to-end machine learning project for classifying user-generated comments into predefined categories using Natural Language Processing, feature engineering, feature selection, and supervised learning.

> **Best verified validation accuracy: 90.64%**

## What This Project Demonstrates

- NLP feature engineering with TF-IDF unigrams and bigrams
- Sparse high-dimensional feature selection with chi-square statistics
- Fusion of text features with numerical and categorical metadata
- Handling class imbalance with class-weighted Logistic Regression
- Comparative evaluation of multiple classical ML approaches
- Reproducible validation using a stratified split and fixed random state

## Dataset

The original Kaggle environment contains:

```text
Sample.csv
train.csv
test.csv
```

The dataset is **not included** in this repository.

| Split | Rows | Columns |
|---|---:|---:|
| Train | 198,000 | 15 |
| Test | 102,000 | 14 |

The training data combines comment text with numerical and categorical metadata.

## Methodology

```text
                         ┌─────────────────────┐
                         │      Raw Data       │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
          ┌───────────────┐                   ┌────────────────┐
          │ Comment Text  │                   │ Structured Data│
          └───────┬───────┘                   └───────┬────────┘
                  │                                   │
                  ▼                                   ▼
          TF-IDF: 20,000                   Scaling / One-Hot
                  │                                   │
                  ▼                                   │
       SelectKBest + Chi-square                       │
                  │                                   │
                  ▼                                   │
          5,000 text features                         │
                  │                                   │
                  └──────────────┬────────────────────┘
                                 ▼
                         Feature Fusion
                                 │
                                 ▼
                Class-Weighted Logistic Regression
                                 │
                                 ▼
                         Final Prediction
```

### Text Representation

```python
TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
```

Unigrams capture individual terms while bigrams capture short multi-word patterns.

### Numerical Features

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

Numerical features are standardized with `StandardScaler` before feature fusion.

### Categorical Features

```text
race
religion
gender
```

These features are encoded with:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)
```

### Feature Selection

The 20,000-dimensional TF-IDF representation is reduced to 5,000 features using:

```python
SelectKBest(
    score_func=chi2,
    k=5000
)
```

This reduces the text feature space by **75%** before fusion with structured features.

## Models Explored

The notebook evaluates several approaches, including:

- Logistic Regression
- Multinomial Naive Bayes
- LinearSVC
- Logistic Regression with feature fusion
- Feature selection + numerical feature fusion
- Voting Classifier
- Hyperparameter-tuned Logistic Regression
- Truncated SVD as an alternative dimensionality-reduction experiment

The strongest verified configuration is **class-weighted Logistic Regression** using selected TF-IDF features with numerical feature fusion.

## Results

| Metric | Result |
|---|---:|
| Training samples | 198,000 |
| Test samples | 102,000 |
| Initial TF-IDF features | 20,000 |
| Selected TF-IDF features | 5,000 |
| Feature-space reduction | 75% |
| Best verified validation accuracy | **90.64%** |
| Final model | Logistic Regression |
| Feature selection | SelectKBest + chi-square |
| Validation split | Stratified |
| Random state | 42 |

Detailed experiments and notebook outputs are available in `nlp_comment_classification.ipynb`.

> **Important:** 90.64% is validation accuracy. It is **not** being presented as test-set accuracy or as a Kaggle leaderboard score.

Detailed configuration is documented in [`results/model_results.md`](results/model_results.md).

## Class Imbalance

The final Logistic Regression configuration uses:

```python
{
    0: 1,
    1: 3,
    2: 2,
    3: 5
}
```

The weighting gives greater training importance to selected minority classes.

## Repository Structure

```text
nlp-comment-classification/
│
├── README.md
├── nlp_comment_classification.ipynb
├── requirements.txt
├── LICENSE
├── .gitignore
└── results/
    └── model_results.md
```

## Installation

```bash
git clone https://github.com/AlakhyaIITM/nlp-comment-classification.git
cd nlp-comment-classification

python -m venv .venv
```

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

## Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
nlp_comment_classification.ipynb
```

### Dataset Path

The notebook was developed in Kaggle and expects:

```python
DATA_PATH = "/kaggle/input"
```

For local execution, obtain the permitted dataset files and update `DATA_PATH` to the local dataset directory.

## Kaggle

The project was originally developed and executed on Kaggle.

**Kaggle Notebook:**

https://www.kaggle.com/code/alakhyasarkar/23f1001050-notebook-t12026

## Reproducibility Notes

- The validation split is stratified.
- `random_state=42` is used for the documented validation setup.
- The dataset is not redistributed with this repository.
- No trained model artifact is committed; the notebook contains the training workflow.
- The current implementation is Kaggle-oriented rather than packaged as a standalone application.

## Limitations & Responsible Use

The dataset contains demographic attributes such as race, religion, and gender. These variables can introduce fairness, privacy, and representational concerns. Model performance should therefore not be interpreted as evidence that the classifier is unbiased or suitable for high-stakes automated decisions.

Other limitations include:

- Only the documented validation result is claimed in this repository.
- The test-set and leaderboard performance are not reported here.
- The original dataset is not included.
- The notebook is not currently packaged as a production inference pipeline.

## Future Improvements

- Report precision, recall, and F1 by class
- Add a confusion matrix and class-level error analysis
- Compare word-level and character-level features
- Perform systematic hyperparameter optimization
- Evaluate transformer-based text representations
- Investigate more robust imbalance-handling strategies
- Package the final pipeline for reproducible inference
- Build an inference API or interactive prediction interface

## Author

**Alakhya Sarkar**  
B.S. in Data Science and Applications  
Indian Institute of Technology Madras

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

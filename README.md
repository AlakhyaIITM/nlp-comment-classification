# NLP Comment Classification System

An end-to-end machine learning project for classifying user-generated comments into predefined categories using Natural Language Processing, feature engineering, feature selection, and supervised learning.

> **Best validation accuracy: 90.64%**

## Project Highlights

- 198,000 training samples and 102,000 test samples
- TF-IDF with unigrams and bigrams
- 20,000 initial text features reduced to 5,000 using SelectKBest + chi-square
- Numerical metadata integrated with text features
- Categorical metadata encoded with OneHotEncoder
- Class imbalance addressed with class-weighted Logistic Regression
- Multiple models and feature configurations evaluated
- Stratified train/validation split with `random_state=42`

## Dataset

The original Kaggle environment contains:

```text
Sample.csv
train.csv
test.csv
```

The dataset is **not included** in this repository.

The training data contains 198,000 rows and 15 columns; the test data contains 102,000 rows and 14 columns. The data combines comment text with numerical and categorical metadata.

## Methodology

```text
Raw Data
   │
   ├── Text ──────────────► TF-IDF (20,000 features)
   │                              │
   │                              ▼
   │                     SelectKBest + Chi-square
   │                              │
   │                              ▼
   │                       5,000 selected features
   │
   └── Structured Data ──► Scaling / One-Hot Encoding
                                  │
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

The use of unigrams and bigrams allows the model to capture individual terms as well as short multi-word patterns.

### Numerical Features

The notebook uses:

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

The notebook encodes:

```text
race
religion
gender
```

using:

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

This represents a **75% reduction** in the TF-IDF feature space.

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

The final selected approach is **class-weighted Logistic Regression** using selected TF-IDF features with numerical feature fusion.

## Results

| Metric | Result |
|---|---:|
| Training samples | 198,000 |
| Test samples | 102,000 |
| Initial TF-IDF features | 20,000 |
| Selected TF-IDF features | 5,000 |
| TF-IDF feature reduction | 75% |
| Best validation accuracy | **90.64%** |
| Final model | Logistic Regression |
| Feature selection | SelectKBest + chi-square |
| Validation split | Stratified |
| Random state | 42 |

The detailed experiments and outputs are available in the notebook.

> **Important:** 90.64% is validation accuracy. It is not being presented as test-set accuracy or as a Kaggle leaderboard score.

## Class Imbalance

The final Logistic Regression configuration uses class weights:

```python
{
    0: 1,
    1: 3,
    2: 2,
    3: 5
}
```

This gives greater training importance to selected minority classes.

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

Clone the repository:

```bash
git clone https://github.com/AlakhyaIITM/nlp-comment-classification.git
cd nlp-comment-classification
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
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

The notebook was developed in Kaggle and expects the dataset under:

```python
DATA_PATH = "/kaggle/input"
```

Therefore, it is directly reproducible in its original Kaggle environment. To run locally, download the permitted dataset files and update `DATA_PATH` to the local dataset directory.

## Kaggle

The project was originally developed and executed on Kaggle.

**Kaggle Notebook:**

https://www.kaggle.com/code/alakhyasarkar/23f1001050-notebook-t12026

## Limitations

- The dataset is not redistributed with this repository.
- The reported 90.64% figure is validation accuracy, not test-set accuracy.
- The current notebook is Kaggle-oriented rather than packaged as a standalone application.
- No trained model artifact is included; the notebook contains the training workflow.

## Future Improvements

- Evaluate transformer-based text representations
- Add word- and character-level features
- Perform systematic hyperparameter optimization
- Report precision, recall, and F1 by class
- Investigate more robust imbalance-handling strategies
- Build an inference API or interactive prediction interface
- Package the final pipeline for deployment

## Author

**Alakhya Sarkar**  
B.S. in Data Science and Applications  
Indian Institute of Technology Madras

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

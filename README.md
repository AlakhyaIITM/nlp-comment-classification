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
- Dataset included in the repository as split CSV files to keep individual files below GitHub's browser upload limit

## Dataset

The project dataset contains comment text together with numerical and categorical metadata.

The complete dataset is stored in the [`data/`](data/) directory. Because the original `train.csv` and `test.csv` files were too large for convenient GitHub browser uploads, they are split into multiple CSV files without removing or changing rows.

```text
data/
├── train_part_1.csv
├── train_part_2.csv
├── train_part_3.csv
├── train_part_4.csv
├── test_part_1.csv
├── test_part_2.csv
└── Sample.csv
```

The complete datasets contain:

- Training data: **198,000 rows, 15 columns**
- Test data: **102,000 rows, 14 columns**
- Submission sample: **102,000 rows**

The training columns include comment text, numerical metadata, categorical metadata, and the target `label`.

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

using `OneHotEncoder` with unknown-category handling.

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

Detailed experiments and outputs are available in [`nlp_comment_classification.ipynb`](nlp_comment_classification.ipynb), while the final configuration is summarized in [`results/model_results.md`](results/model_results.md).

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
├── data/
│   ├── train_part_1.csv
│   ├── train_part_2.csv
│   ├── train_part_3.csv
│   ├── train_part_4.csv
│   ├── test_part_1.csv
│   ├── test_part_2.csv
│   ├── Sample.csv
│   └── README.md
├── results/
│   └── model_results.md
├── README.md
├── nlp_comment_classification.ipynb
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/AlakhyaIITM/nlp-comment-classification.git
cd nlp-comment-classification
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open `nlp_comment_classification.ipynb`.

## Reproducibility

The main validation experiments use a stratified train/validation split with `random_state=42`. The notebook contains the complete preprocessing, feature engineering, model training, and evaluation workflow.

## Responsible Use

This project is intended as a machine learning and NLP portfolio project. Because the dataset includes demographic attributes such as race, religion, gender, and disability, model outputs should be interpreted carefully. High validation accuracy does not by itself establish fairness, safety, or suitability for real-world moderation decisions.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

# Model Results

## Best Verified Result

| Final configuration | Validation accuracy |
|---|---:|
| Class-weighted Logistic Regression + selected TF-IDF + numerical feature fusion | **90.64%** |

> **Evaluation note:** 90.64% is the notebook's reported validation accuracy. It is **not** test-set accuracy and is **not** a Kaggle leaderboard score.

## Final Configuration

| Component | Configuration |
|---|---|
| Text representation | TF-IDF, unigrams + bigrams |
| Initial text features | 20,000 |
| Feature selection | SelectKBest + chi-square |
| Selected text features | 5,000 |
| Numerical features | 8 |
| Categorical features | `race`, `religion`, `gender` |
| Numerical preprocessing | StandardScaler |
| Categorical preprocessing | OneHotEncoder (`handle_unknown="ignore"`) |
| Class weights | `{0: 1, 1: 3, 2: 2, 3: 5}` |
| Validation strategy | Stratified train/validation split |
| Random state | 42 |

## Feature Groups

### Text

The comment field is represented using TF-IDF with:

```python
TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
```

### Numerical

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

### Categorical

```text
race
religion
gender
```

## Feature Reduction

The text representation is reduced from 20,000 to 5,000 TF-IDF features using chi-square feature selection — a **75% reduction** in the text feature space before fusion with structured features.

## Interpretation

The selected pipeline combines sparse lexical information from comments with structured metadata and class weighting. This produced the strongest **verified validation result currently documented in the repository**.

The notebook contains the underlying experiments, training workflow, and prediction generation steps.

## Reproducibility

The project was developed in Kaggle using the original competition dataset. The dataset itself is intentionally not committed to this repository. To reproduce the experiment, obtain the permitted dataset files and update the notebook's `DATA_PATH` for the execution environment.

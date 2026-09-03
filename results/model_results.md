# Model Results

## Final Result

The best validation result reported by the notebook is:

| Configuration | Validation Accuracy |
|---|---:|
| Class-weighted Logistic Regression with selected TF-IDF + numerical feature fusion | **90.64%** |

## Final Configuration

- Text representation: TF-IDF
- TF-IDF features before selection: 20,000
- Selected text features: 5,000
- Feature selection: SelectKBest + chi-square
- Numerical features: 8
- Categorical features: race, religion, gender
- Numerical preprocessing: StandardScaler
- Categorical preprocessing: OneHotEncoder
- Class weights: `{0: 1, 1: 3, 2: 2, 3: 5}`
- Validation strategy: stratified train/validation split
- Random state: 42

## Important Note

The 90.64% figure is validation accuracy. It should not be presented as test-set accuracy or as a Kaggle leaderboard score.

The notebook contains the detailed experiments and their outputs.

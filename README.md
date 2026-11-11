# Career Outcome Prediction Using Learning Behavior Data

Machine Learning Internship Capstone Project — Schoolliant

## Overview

This project asks whether a student's in-course performance (final score, project score, soft skill score, attendance, mock interview score) can predict whether they get placed in a job, and, as a secondary question, which salary band they land in.

**Short version of the findings**

- Placement is almost perfectly predictable from these five scores (about 98.8% cross-validated accuracy against a 59.5% majority-class baseline). The models are statistically tied, so no single model can be called "best".
- Salary band is not predictable from these features. Models score below the majority-class baseline.
- The dataset is synthetic assignment data, and the diagnostics in the notebook show it. The 98.8% is therefore a ceiling on a clean toy problem, not an estimate of real-world performance.

## Repository Contents

| File | Description |
|---|---|
| `projectid6_v2.ipynb` | Revised notebook (current): baseline comparison, repeated CV, corrected feature importance, nested hyperparameter tuning, salary and course checks, synthetic-data diagnostics, final pipeline |
| `ProjectID6.ipynb` | Original notebook submitted to Schoolliant, kept for reference |
| `README.md` | This file |

**Note:** The dataset (`schoolliant_career_outcome_prediction.csv`) is not included because it comes from Schoolliant's course records. To run the notebook, add a copy of the file and edit `CSV_PATH` in the first cell.

## What Changed From the Original Submission

The original report made several claims that did not hold up under a more careful analysis. They are corrected here.

| Original claim | What the revised analysis shows |
|---|---|
| Logistic Regression is the best and most stable model | Logistic Regression, SVM and Random Forest are within about 0.2 points of each other, against fold-to-fold noise of about 1.2 points. They are tied. |
| Final score is the weakest predictor | Not supported. It was an artifact of Random Forest's impurity-based importance. Alone, final score has an AUC of 0.905 (others 0.92–0.94), and it has the largest coefficient in the Logistic Regression. The ranking changes depending on the method used. |
| `interview_readiness_score` outperforms its components | No performance gain. Accuracy changes by about 0.04 points, which is noise. The earlier claim rested on importance shares, which are not a measure of performance. |
| Salary prediction fails because of class imbalance | Salary band shows no relationship to the features (no significant ANOVA result), so more Senior-band examples would not fix it. |
| Random Forest tuning improved stability | Nested cross-validation shows tuning gains of 0.1–0.9 points, within noise. |

## Approach

1. **Cleaning** — Dropped 4 duplicate rows (404 → 400). The 12 missing `mock_interview_score` values are imputed with the training-fold mean inside each pipeline, which avoids leakage.
2. **Models** — Majority-class baseline, Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting. KNN, Naive Bayes and Decision Tree from the original were dropped because they tied or trailed.
3. **Evaluation** — 5-fold stratified CV repeated 20 times, reporting accuracy, ROC-AUC, and recall for both classes. The original 80/20 hold-out split (seed 42) is kept for comparability.
4. **Feature importance** — Random Forest impurity importance, permutation importance (computed on held-out folds), and standardized Logistic Regression coefficients, side by side.
5. **Extra checks** — `interview_readiness_score`, `course_name` as a feature, and salary band against its majority-class baseline.
6. **Tuning** — Grid search with nested CV for all four models, so tuned scores are not optimistic.
7. **Synthetic-data diagnostics** — Within-class correlations, value caps and floors, and score distributions.
8. **Final pipeline** — Imputer, scaler and Logistic Regression saved with `joblib` (`placement_model.joblib`).

## Key Results

Repeated 5-fold CV, 20 repeats (mean ± std across folds):

| Model | Accuracy | ROC-AUC | Recall (not placed) |
|---|---|---|---|
| Majority baseline | 59.50% ± 0.61 | 0.500 | 0.00% |
| Logistic Regression | 98.79% ± 1.14 | 0.9982 | 97.96% |
| SVM (RBF) | 98.87% ± 1.17 | 0.9994 | 98.15% |
| Random Forest | 98.65% ± 1.21 | 0.9992 | 98.18% |
| Gradient Boosting | 97.66% ± 2.12 | 0.9956 | 96.45% |

Hyperparameter tuning (nested CV, 5 folds × 2 repeats):

| Model | Untuned | Tuned |
|---|---|---|
| Logistic Regression | 98.88% | 99.12% |
| SVM (RBF) | 99.00% | 99.25% |
| Random Forest | 98.50% | 98.62% |
| Gradient Boosting | 97.75% | 98.62% |

All differences between models are within fold-to-fold noise. Logistic Regression remains a reasonable choice because it is simple to explain, not because it is measurably better.

**Other results**

- **Salary band:** best model 50.9% accuracy vs. a 54.2% majority-class baseline.
- **Course:** placement rates range from 47% to 76% across the seven courses, but the difference is not significant at the 5% level (chi-square p = 0.064), and adding `course_name` does not improve accuracy.
- **Importance:** no feature clearly dominates. Single-feature AUCs range from 0.905 (final score) to 0.943 (project score, mock interview score).

## Evidence the Data Is Synthetic

- Correlation between scores overall is about 0.5, but within each placement class it is only about 0.05 (placed) and 0.09 (not placed). Scores look independent once the label is known.
- Not-placed students are capped at exactly 95.0 on final, project and soft skill scores, while placed students have hard floors (for example attendance never below 68.8).
- Near-perfect separation from five scores on 400 rows is unusual for real placement outcomes, which depend on many factors outside course performance.

## Limitations

- **Synthetic data.** The accuracy figures describe this dataset only and should not be read as real-world performance. Feature importance reflects how the data was generated, not real hiring behavior.
- **Small sample.** 400 rows, 80 in the hold-out set (one student is 1.25 accuracy points).
- **End-of-course features only.** An early-warning version would need mid-course data.
- **Overconfident probabilities.** Predicted probabilities are close to 0 or 1 because the classes barely overlap. They are not calibrated estimates.
- **Fold overlap.** The ± values come from overlapping CV folds and understate true uncertainty, so models are described as "tied", not statistically compared.

## Next Steps

- Validate on real data. The Open University Learning Analytics Dataset (OULAD) is a candidate for a mid-course early-warning version.
- Collect real placement outcomes, including salary, before attempting any salary model.

## Tech Stack

- Python 3, pandas, NumPy, SciPy
- scikit-learn (Logistic Regression, SVM, Random Forest, Gradient Boosting, pipelines, CV, grid search)
- Matplotlib, joblib
- Jupyter / Google Colab

## Running the Notebook

1. Clone this repository.
2. Open `projectid6_v2.ipynb` in Jupyter or Google Colab.
3. Add the dataset and edit `CSV_PATH` in the first cell.
4. Run all cells top to bottom (the full run takes about 5 minutes; the tuning cell is the slowest).

## Author

Syed Abdul Raheem

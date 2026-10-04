# Career Outcome Prediction Using Learning Behavior Data

Machine Learning Internship Capstone Project — Schoolliant

## Overview

This project looks at whether a student's in-course performance can predict what happens to them afterward — specifically, whether they get placed in a job, and secondarily, what salary band they're likely to land in. The data comes from Schoolliant's course records: final score, project score, soft skill score, attendance, and mock interview performance for 404 students.

The short answer: placement status turns out to be very predictable from this data (99%+ cross-validated accuracy with Logistic Regression). Salary band is a much harder problem — the dataset simply doesn't have enough Senior-band examples for any model to learn that pattern reliably.

## Repository Contents

| File | Description |
|---|---|
| `ProjectID6.ipynb` | Full notebook — data cleaning, baseline models, tree/ensemble models, feature engineering, cross-validation, and hyperparameter tuning |
| `README.md` | This file |

**Note:** The dataset (`schoolliant_career_outcome_prediction.csv`) is not included in this repository, as it contains Schoolliant's course and student records and is not intended for public distribution. To run the notebook, place a copy of the dataset in the same directory and update the file path in the first cell as needed.

## Approach

1. **Data Cleaning** — Imputed 12 missing `mock_interview_score` values with the column mean, dropped 4 duplicate rows (404 → 400 records).
2. **Baseline Models** — Logistic Regression, KNN, and Naive Bayes on placement status (placed / not placed).
3. **Trees & Ensembles** — Decision Tree, Random Forest, SVM, and Gradient Boosting, compared against the baselines.
4. **Feature Engineering** — Combined `mock_interview_score` and `soft_skill_score` into a single `interview_readiness_score`, which became the single strongest predictor of placement. Also explored salary band as a secondary, multi-class target.
5. **Cross-Validation & Tuning** — 5-fold stratified cross-validation across all models, plus GridSearchCV on Random Forest.

## Key Results

| Model | CV Accuracy (mean) | Std. Dev. |
|---|---|---|
| Logistic Regression | 99.06% | 0.77% |
| Naive Bayes / SVM | 98.75% | 1.82% |
| KNN | 98.44% | 1.40% |
| Random Forest | 98.12% | 1.82% |
| Gradient Boosting | 96.25% | 2.12% |
| Decision Tree | 95.63% | 2.86% |

**Recommended model:** Logistic Regression — highest and most stable cross-validated accuracy, and straightforward to explain to a non-technical audience.

**Top predictors of placement:** `interview_readiness_score` (engineered), attendance %, and project score — together accounting for the large majority of model importance. Final exam score, somewhat surprisingly, was the weakest predictor.

**Known limitation:** Salary band prediction (Entry / Mid / Senior) performs poorly (43.75% accuracy) due to severe class imbalance — only 22 Senior-band students in the dataset.

## Tech Stack

- Python 3, pandas, NumPy
- scikit-learn (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, SVM, Gradient Boosting)
- Jupyter / Google Colab

## Running the Notebook

1. Clone this repository.
2. Open `ProjectID6.ipynb` in Jupyter or Google Colab.
3. Add the dataset to the same directory (or update the file path in the first cell) — see note above.
4. Run all cells top to bottom.

## Author

Abdul Raheem

# Amazon Prime Customer Churn Prediction

This project builds a supervised machine learning pipeline to predict whether an Amazon Prime customer is likely to churn. It uses synthetic, preprocessed Amazon Prime-style customer data, Pandas, Scikit-learn, Logistic Regression, Decision Trees, hyperparameter tuning, threshold analysis, EDA reports, and SQL/SQLite retention analysis.

## Project Structure

```text
amazon_prime_churn_prediction/
  data/preprocessed/amazon_prime_churn_preprocessed.csv
  data/amazon_prime_churn.db                 # generated after SQL load
  models/amazon_prime_churn_model.joblib      # generated after training
  reports/
    eda_numeric_summary.csv
    eda_churn_by_plan.csv
    eda_churn_by_engagement.csv
    model_comparison.csv
    model_metrics.json
    confusion_matrix.csv
    hyperparameter_tuning.csv
    threshold_scan.csv
    top_retention_targets.csv
    decision_tree_feature_importance.csv
    logistic_regression_coefficients.csv
  sql/
    schema.sql
    analysis_queries.sql
  src/
    generate_preprocessed_data.py
    eda.py
    load_to_sqlite.py
    train_model.py
    predict.py
    config.py
  requirements.txt
```

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/generate_preprocessed_data.py
python src/eda.py
python src/load_to_sqlite.py
python src/train_model.py
python src/predict.py
```

On Windows, you can also run:

```bat
run_project.bat
```

## Dataset

The included CSV is a small preprocessed sample so the project can be inspected quickly. The generator script recreates the full 1,200-row synthetic dataset before analysis and training. This avoids pretending the data is a real Amazon dataset while still giving a complete supervised ML workflow.

Target column:

- `churn`: `1` means the customer churned, `0` means retained.

## Models

The training script compares:

- Logistic Regression
- Decision Tree Classifier

Both models use `GridSearchCV` hyperparameter tuning. The best model is selected by F1 score, then ROC-AUC, then recall. The script also scans probability thresholds so the retention team can choose the right trade-off between recall and precision.

Latest verified result:

- Best model: Logistic Regression
- Accuracy: `0.7417`
- Precision: `0.3614`
- Recall: `0.7692`
- F1 score: `0.4918`
- ROC-AUC: `0.8307`

Precision is intentionally reported, not hidden. Churn datasets are usually imbalanced, so the project prioritizes finding at-risk customers while documenting the false-positive trade-off.

## EDA Findings

- Monthly-plan customers have higher churn than annual-plan customers.
- Very-low-engagement customers are the highest-risk engagement group.
- Inactivity, support tickets, payment failures, lower ratings, and shorter tenure are strong churn signals.

## Retention Use Case

Customers with high churn probability can be prioritized for:

- Subscription discounts or annual plan offers
- Payment failure support
- Win-back email campaigns
- Customer service follow-up after repeated support tickets
- Personalized recommendations for low-engagement accounts

## SQL

Use `sql/schema.sql` to create a table for the preprocessed dataset and `sql/analysis_queries.sql` to inspect churn rate, plan-level risk, engagement, and high-risk retention segments. The `src/load_to_sqlite.py` script loads the dataset into a local SQLite database for a runnable SQL workflow.

## Major Project Readiness

This is now suitable as a major project base because it includes data generation, preprocessing-ready data, EDA, SQL integration, tuned supervised ML models, model evaluation, explainability reports, retention targeting, and prediction inference. For a real company-grade version, replace the synthetic dataset with approved production or survey data.

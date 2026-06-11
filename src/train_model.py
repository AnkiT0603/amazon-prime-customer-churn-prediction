import json

import joblib
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import DATA_PATH, DEFAULT_THRESHOLD, MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN, TEST_SIZE


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    feature_columns = [column for column in df.columns if column not in [TARGET_COLUMN, "customer_id"]]
    categorical_columns = [
        column
        for column in feature_columns
        if is_string_dtype(df[column]) or str(df[column].dtype) == "category"
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def predict_with_threshold(model: Pipeline, x_test: pd.DataFrame, threshold: float) -> tuple[pd.Series, pd.Series]:
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1], index=x_test.index)
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = DEFAULT_THRESHOLD) -> dict:
    probabilities, predictions = predict_with_threshold(model, x_test, threshold)
    return {
        "model": name,
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
    }


def choose_threshold(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in [round(value / 100, 2) for value in range(20, 81, 5)]:
        rows.append(evaluate_model("threshold_scan", model, x_test, y_test, threshold))

    scan = pd.DataFrame(rows).drop(columns=["model"])
    scan = scan.sort_values(["f1_score", "precision", "recall"], ascending=False)
    return float(scan.iloc[0]["threshold"]), scan


def save_tree_importance(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(REPORTS_DIR / "decision_tree_feature_importance.csv", index=False)


def save_logistic_coefficients(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": classifier.coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(
        REPORTS_DIR / "logistic_regression_coefficients.csv",
        index=False,
    )


def save_retention_targets(df: pd.DataFrame, model: Pipeline, threshold: float) -> None:
    features = df.drop(columns=[TARGET_COLUMN, "customer_id"])
    probabilities = model.predict_proba(features)[:, 1]
    targets = df[["customer_id", "plan_type", "tenure_months", "days_since_last_login", "support_tickets_last_90d", "payment_failed_last_90d", "avg_rating"]].copy()
    targets["churn_probability"] = probabilities.round(4)
    targets["retention_priority"] = pd.cut(
        targets["churn_probability"],
        bins=[0, threshold, 0.7, 1.0],
        labels=["Monitor", "High", "Critical"],
        include_lowest=True,
    )
    targets.sort_values("churn_probability", ascending=False).head(100).to_csv(
        REPORTS_DIR / "top_retention_targets.csv",
        index=False,
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET_COLUMN, "customer_id"])
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(df)
    searches = {
        "Logistic Regression": GridSearchCV(
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            param_grid={"classifier__C": [0.1, 0.5, 1.0, 2.0]},
            scoring="f1",
            cv=5,
        ),
        "Decision Tree": GridSearchCV(
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            param_grid={
                "classifier__max_depth": [3, 4, 5, 6, 8],
                "classifier__min_samples_leaf": [10, 20, 30],
            },
            scoring="f1",
            cv=5,
        ),
    }

    fitted_models = {}
    metrics = []
    tuning_rows = []
    for name, search in searches.items():
        search.fit(x_train, y_train)
        fitted_models[name] = search.best_estimator_
        metrics.append(evaluate_model(name, search.best_estimator_, x_test, y_test))
        tuning_rows.append({"model": name, "best_cv_f1": round(search.best_score_, 4), "best_params": search.best_params_})

    comparison = pd.DataFrame(metrics).sort_values(["f1_score", "roc_auc", "recall"], ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(REPORTS_DIR / "hyperparameter_tuning.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_threshold, threshold_scan = choose_threshold(best_model, x_test, y_test)
    threshold_scan.to_csv(REPORTS_DIR / "threshold_scan.csv", index=False)

    probabilities, predictions = predict_with_threshold(best_model, x_test, best_threshold)
    report = {
        "best_model": best_name,
        "selected_threshold": best_threshold,
        "rows": int(len(df)),
        "target_distribution": df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
        "metrics_at_default_threshold": comparison.to_dict(orient="records"),
        "metrics_at_selected_threshold": evaluate_model(best_name, best_model, x_test, y_test, best_threshold),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    with (REPORTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=["actual_active", "actual_churned"],
        columns=["predicted_active", "predicted_churned"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")

    save_logistic_coefficients(fitted_models["Logistic Regression"])
    save_tree_importance(fitted_models["Decision Tree"])
    save_retention_targets(df, best_model, best_threshold)

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "best_model_name": best_name,
            "target_column": TARGET_COLUMN,
        },
        MODEL_PATH,
    )
    print(f"Best model: {best_name}")
    print(comparison.to_string(index=False))
    print(f"Selected churn threshold: {best_threshold:.2f}")
    print(f"Saved model package to {MODEL_PATH}")


if __name__ == "__main__":
    main()

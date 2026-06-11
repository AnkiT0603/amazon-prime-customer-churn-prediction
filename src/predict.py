import json

import joblib
import pandas as pd

from config import MODEL_PATH


SAMPLE_CUSTOMER = {
    "age": 31,
    "tenure_months": 5,
    "monthly_fee": 14.99,
    "plan_type": "Monthly",
    "watch_hours_per_week": 1.7,
    "days_since_last_login": 38,
    "support_tickets_last_90d": 2,
    "payment_failed_last_90d": 1,
    "discount_used": 0,
    "avg_rating": 2.6,
    "device_type": "Mobile",
    "region": "West",
    "payment_method": "Credit Card",
}


def main() -> None:
    model_package = joblib.load(MODEL_PATH)
    model = model_package["model"] if isinstance(model_package, dict) else model_package
    threshold = model_package.get("threshold", 0.5) if isinstance(model_package, dict) else 0.5
    customer = pd.DataFrame([SAMPLE_CUSTOMER])
    churn_probability = model.predict_proba(customer)[0][1]
    prediction = int(churn_probability >= threshold)

    print(
        json.dumps(
            {
                "prediction": "Churn" if prediction else "Retain",
                "churn_probability": round(float(churn_probability), 4),
                "threshold": round(float(threshold), 2),
                "customer": SAMPLE_CUSTOMER,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

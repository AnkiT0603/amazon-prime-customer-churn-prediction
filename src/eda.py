import pandas as pd

from config import DATA_PATH, REPORTS_DIR, TARGET_COLUMN


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    numeric_summary = df.describe().T
    numeric_summary.to_csv(REPORTS_DIR / "eda_numeric_summary.csv")

    churn_by_plan = (
        df.groupby("plan_type")[TARGET_COLUMN]
        .agg(customers="count", churn_rate="mean")
        .assign(churn_rate_pct=lambda data: (data["churn_rate"] * 100).round(2))
        .drop(columns=["churn_rate"])
        .reset_index()
    )
    churn_by_plan.to_csv(REPORTS_DIR / "eda_churn_by_plan.csv", index=False)

    churn_by_engagement = (
        df.assign(
            engagement_band=pd.cut(
                df["watch_hours_per_week"],
                bins=[-1, 2, 6, 12, 100],
                labels=["Very Low", "Low", "Medium", "High"],
            )
        )
        .groupby("engagement_band", observed=True)[TARGET_COLUMN]
        .agg(customers="count", churn_rate="mean")
        .assign(churn_rate_pct=lambda data: (data["churn_rate"] * 100).round(2))
        .drop(columns=["churn_rate"])
        .reset_index()
    )
    churn_by_engagement.to_csv(REPORTS_DIR / "eda_churn_by_engagement.csv", index=False)

    print("Saved EDA summaries to reports/")


if __name__ == "__main__":
    main()

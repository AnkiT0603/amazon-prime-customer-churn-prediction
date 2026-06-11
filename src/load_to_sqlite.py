import sqlite3

import pandas as pd

from config import DATA_PATH, DATABASE_PATH


def main() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    with sqlite3.connect(DATABASE_PATH) as connection:
        df.to_sql("amazon_prime_customers", connection, if_exists="replace", index=False)

        churn_by_plan = pd.read_sql_query(
            """
            SELECT
                plan_type,
                COUNT(*) AS customers,
                ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
            FROM amazon_prime_customers
            GROUP BY plan_type
            ORDER BY churn_rate_pct DESC
            """,
            connection,
        )

    print(f"Loaded {len(df)} rows into {DATABASE_PATH}")
    print(churn_by_plan.to_string(index=False))


if __name__ == "__main__":
    main()

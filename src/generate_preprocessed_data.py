import csv
import random

from config import DATA_PATH, RANDOM_STATE


def sigmoid(value: float) -> float:
    return 1 / (1 + pow(2.718281828, -value))


def main(rows: int = 1200) -> None:
    random.seed(RANDOM_STATE)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    plans = ["Monthly", "Annual"]
    devices = ["Mobile", "TV", "Desktop", "Tablet"]
    regions = ["North", "South", "East", "West", "Central"]
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking"]

    fields = [
        "customer_id",
        "age",
        "tenure_months",
        "monthly_fee",
        "plan_type",
        "watch_hours_per_week",
        "days_since_last_login",
        "support_tickets_last_90d",
        "payment_failed_last_90d",
        "discount_used",
        "avg_rating",
        "device_type",
        "region",
        "payment_method",
        "churn",
    ]

    with DATA_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for idx in range(1, rows + 1):
            age = random.randint(18, 70)
            tenure = random.randint(1, 72)
            plan = random.choices(plans, weights=[0.58, 0.42])[0]
            monthly_fee = 14.99 if plan == "Monthly" else 10.99
            watch_hours = max(0, round(random.gauss(8.0 if plan == "Annual" else 6.5, 4.0), 1))
            inactive_days = random.randint(0, 90)
            tickets = random.choices([0, 1, 2, 3, 4, 5], weights=[55, 22, 11, 7, 3, 2])[0]
            payment_failed = random.choices([0, 1], weights=[86, 14])[0]
            discount_used = random.choices([0, 1], weights=[62, 38])[0]
            rating = min(5, max(1, round(random.gauss(4.0, 0.75), 1)))
            device = random.choice(devices)
            region = random.choice(regions)
            payment_method = random.choice(payment_methods)

            risk = (
                -2.2
                + (inactive_days / 30) * 0.85
                + tickets * 0.35
                + payment_failed * 1.15
                - tenure * 0.025
                - watch_hours * 0.075
                - (rating - 3.0) * 0.55
                + (0.35 if plan == "Monthly" else -0.25)
                + (0.18 if discount_used else 0)
            )
            churn = 1 if random.random() < sigmoid(risk) else 0

            writer.writerow(
                {
                    "customer_id": f"AP{idx:06d}",
                    "age": age,
                    "tenure_months": tenure,
                    "monthly_fee": monthly_fee,
                    "plan_type": plan,
                    "watch_hours_per_week": watch_hours,
                    "days_since_last_login": inactive_days,
                    "support_tickets_last_90d": tickets,
                    "payment_failed_last_90d": payment_failed,
                    "discount_used": discount_used,
                    "avg_rating": rating,
                    "device_type": device,
                    "region": region,
                    "payment_method": payment_method,
                    "churn": churn,
                }
            )

    print(f"Generated {rows} preprocessed rows at {DATA_PATH}")


if __name__ == "__main__":
    main()

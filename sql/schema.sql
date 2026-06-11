CREATE TABLE amazon_prime_customers (
    customer_id VARCHAR(12) PRIMARY KEY,
    age INTEGER NOT NULL,
    tenure_months INTEGER NOT NULL,
    monthly_fee DECIMAL(6, 2) NOT NULL,
    plan_type VARCHAR(20) NOT NULL,
    watch_hours_per_week DECIMAL(5, 2) NOT NULL,
    days_since_last_login INTEGER NOT NULL,
    support_tickets_last_90d INTEGER NOT NULL,
    payment_failed_last_90d INTEGER NOT NULL,
    discount_used INTEGER NOT NULL,
    avg_rating DECIMAL(3, 1) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    region VARCHAR(20) NOT NULL,
    payment_method VARCHAR(30) NOT NULL,
    churn INTEGER NOT NULL CHECK (churn IN (0, 1))
);

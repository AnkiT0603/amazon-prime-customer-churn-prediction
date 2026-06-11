-- Overall churn rate
SELECT
    ROUND(AVG(churn) * 100, 2) AS churn_rate_pct,
    COUNT(*) AS customers
FROM amazon_prime_customers;

-- Churn by subscription plan
SELECT
    plan_type,
    COUNT(*) AS customers,
    ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
FROM amazon_prime_customers
GROUP BY plan_type
ORDER BY churn_rate_pct DESC;

-- High-risk retention segment for outreach
SELECT
    customer_id,
    plan_type,
    days_since_last_login,
    support_tickets_last_90d,
    payment_failed_last_90d,
    avg_rating
FROM amazon_prime_customers
WHERE days_since_last_login >= 30
   OR support_tickets_last_90d >= 2
   OR payment_failed_last_90d = 1
ORDER BY days_since_last_login DESC, support_tickets_last_90d DESC;

-- Engagement and satisfaction profile by churn status
SELECT
    churn,
    ROUND(AVG(watch_hours_per_week), 2) AS avg_watch_hours,
    ROUND(AVG(days_since_last_login), 2) AS avg_days_inactive,
    ROUND(AVG(avg_rating), 2) AS avg_rating
FROM amazon_prime_customers
GROUP BY churn;

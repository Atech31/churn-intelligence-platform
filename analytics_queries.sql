-- 1. Create Base Database Schema
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    signup_date DATE,
    gender VARCHAR(10),
    age INT,
    city_tier VARCHAR(10),
    signup_channel VARCHAR(30)
);

CREATE TABLE transactions (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    order_date DATE,
    order_amount_inr NUMERIC(10, 2),
    category VARCHAR(30),
    payment_method VARCHAR(20),
    is_discount_used INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- 2. RFM Aggregation View
CREATE VIEW v_customer_rfm AS
SELECT 
    c.customer_id,
    c.city_tier,
    c.signup_channel,
    MAX(t.order_date) AS last_order_date,
    COUNT(t.order_id) AS total_orders,
    SUM(t.order_amount_inr) AS total_spend_inr,
    AVG(t.order_amount_inr) AS avg_order_value_inr,
    CAST((JULIANDAY('2026-09-01') - JULIANDAY(MAX(t.order_date))) AS INT) AS recency_days
FROM customers c
LEFT JOIN transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.city_tier, c.signup_channel;

-- 3. High Churn Risk Identification Query
SELECT 
    customer_id,
    city_tier,
    recency_days,
    total_orders,
    total_spend_inr,
    CASE 
        WHEN recency_days > 120 THEN 'High Churn Risk'
        WHEN recency_days BETWEEN 60 AND 120 THEN 'At Risk'
        ELSE 'Active'
    END AS churn_status
FROM v_customer_rfm
ORDER BY recency_days DESC;
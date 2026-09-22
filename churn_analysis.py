import sqlite3
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load datasets
customers = pd.read_csv("customers.csv")
transactions = pd.read_csv("transactions.csv")

# Create SQLite Database in-memory
conn = sqlite3.connect(":memory:")
customers.to_sql("customers", conn, index=False)
transactions.to_sql("transactions", conn, index=False)

# Compute RFM Analysis via SQL
rfm_df = pd.read_sql("""
    SELECT 
        c.customer_id,
        c.age,
        c.city_tier,
        COUNT(t.order_id) AS frequency,
        COALESCE(SUM(t.order_amount_inr), 0) AS monetary,
        CAST((JULIANDAY('2026-09-01') - JULIANDAY(MAX(t.order_date))) AS INT) AS recency
    FROM customers c
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    GROUP BY c.customer_id
""", conn)

# Fill missing recency for non-purchasers
rfm_df['recency'] = rfm_df['recency'].fillna(365)
rfm_df['is_churned'] = (rfm_df['recency'] > 90).astype(int)

# Plot Recency vs Monetary Distribution
fig = px.scatter(
    rfm_df, 
    x='recency', 
    y='monetary', 
    color='is_churned',
    title="Customer Recency vs. Total Spend (Churn Analysis)",
    labels={'recency': 'Days Since Last Order', 'monetary': 'Total Spend (₹)'}
)
fig.write_html("churn_distribution.html")
print("Saved HTML visual report: churn_distribution.html")

# Train Machine Learning Churn Predictor
X = rfm_df[['age', 'frequency', 'monetary', 'recency']]
y = rfm_df['is_churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n--- Model Performance Report ---")
print(classification_report(y_test, y_pred))
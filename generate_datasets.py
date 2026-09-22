import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_customers = 500

# 1. Customers Dataset
customers = pd.DataFrame({
    'customer_id': [f"CUST_{1000 + i}" for i in range(n_customers)],
    'signup_date': [datetime(2025, 1, 1) + timedelta(days=int(np.random.randint(0, 365))) for _ in range(n_customers)],
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'age': np.random.randint(18, 65, n_customers),
    'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_customers, p=[0.4, 0.35, 0.25]),
    'signup_channel': np.random.choice(['Organic Direct', 'Paid Ads', 'Referral', 'Social Media'], n_customers)
})

# 2. Transactions Dataset
orders = []
for cid in customers['customer_id']:
    num_orders = np.random.poisson(lam=4)
    for _ in range(num_orders):
        orders.append({
            'order_id': f"ORD_{np.random.randint(100000, 999999)}",
            'customer_id': cid,
            'order_date': datetime(2025, 6, 1) + timedelta(days=int(np.random.randint(0, 450))),
            'order_amount_inr': round(np.random.exponential(scale=1500) + 200, 2),
            'category': np.random.choice(['Electronics', 'Apparel', 'Home & Kitchen', 'Beauty', 'Books']),
            'payment_method': np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'COD']),
            'is_discount_used': np.random.choice([0, 1], p=[0.6, 0.4])
        })

df_orders = pd.DataFrame(orders)

# Save to CSV
customers.to_csv("customers.csv", index=False)
df_orders.to_csv("transactions.csv", index=False)
print("Data generation complete! Saved customers.csv and transactions.csv")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import sqlite3

st.set_page_config(page_title="Churn & Revenue Intelligence Engine", layout="wide")

st.title("🛒 E-Commerce Customer Retention & Churn Analytics")

# --- DATA GENERATION & LOADING ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_customers = 1000
    customers = pd.DataFrame({
        'customer_id': [f"CUST_{1000 + i}" for i in range(n_customers)],
        'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n_customers, p=[0.4, 0.35, 0.25]),
        'signup_channel': np.random.choice(['Organic Direct', 'Paid Ads', 'Referral', 'Social Media'], n_customers)
    })
    
    orders = []
    for cid in customers['customer_id']:
        for _ in range(np.random.poisson(lam=4)):
            orders.append({
                'order_id': f"ORD_{np.random.randint(100000, 999999)}",
                'customer_id': cid,
                'order_date': (pd.Timestamp('2025-06-01') + pd.Timedelta(days=int(np.random.randint(0, 450)))).strftime('%Y-%m-%d'),
                'order_amount_inr': round(float(np.random.exponential(scale=1500) + 200), 2)
            })
    df_orders = pd.DataFrame(orders)
    
    # Process directly in Pandas to avoid SQLite thread/caching connection errors
    rfm_df = df_orders.groupby('customer_id').agg(
        frequency=('order_id', 'count'),
        monetary=('order_amount_inr', 'sum'),
        last_order=('order_date', 'max')
    ).reset_index()
    
    rfm_df = customers.merge(rfm_df, on='customer_id', how='left')
    rfm_df['frequency'] = rfm_df['frequency'].fillna(0)
    rfm_df['monetary'] = rfm_df['monetary'].fillna(0.0)
    
    ref_date = pd.Timestamp('2026-09-01')
    rfm_df['recency'] = rfm_df['last_order'].apply(
        lambda x: (ref_date - pd.Timestamp(x)).days if pd.notnull(x) else 365
    )
    rfm_df['is_churned'] = (rfm_df['recency'] > 90).astype(int)
    
    return rfm_df

rfm_df = load_data()

# --- MODEL TRAINING ---
X = filtered_df[['frequency', 'monetary', 'recency']]
y = filtered_df['is_churned']

if len(filtered_df) > 10 and len(y.unique()) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
else:
    st.warning("Not enough data points selected to train the model. Please expand your filters.")

# --- PAGE 1: DASHBOARD ---
if page == "📈 Dashboard & Curves":
    st.header("Advanced Curves & Analytics")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(filtered_df, x="recency", y="monetary", color="is_churned", title="1. Recency vs Monetary")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.histogram(filtered_df, x="monetary", color="city_tier", title="2. Monetary Distribution by City Tier")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.box(filtered_df, x="signup_channel", y="frequency", title="3. Purchase Frequency by Channel")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.scatter(filtered_df, x="frequency", y="monetary", color="city_tier", title="4. Frequency vs Monetary")
        st.plotly_chart(fig4, use_container_width=True)
        
        fig5 = px.histogram(filtered_df, x="recency", color="signup_channel", title="5. Recency Distribution by Channel")
        st.plotly_chart(fig5, use_container_width=True)
        
        fig6 = px.pie(filtered_df, names="is_churned", title="6. Overall Churn Ratio")
        st.plotly_chart(fig6, use_container_width=True)

# --- PAGE 2: EXECUTIVE CONTROL PANEL ---
elif page == "🤖 Executive Control Panel":
    st.header("Executive Control Panel")
    with st.expander("🤖 Generate AI Executive Strategy Briefing", expanded=True):
        st.write("### Executive Summary Briefing")
        st.write(f"- **Total Analyzed Cohort**: {len(filtered_df)} customers")
        st.write(f"- **High-Risk Churn Rate**: {round((filtered_df['is_churned'].sum() / max(len(filtered_df), 1)) * 100, 2)}%")
        st.write(f"- **Total At-Risk Revenue**: ₹{filtered_df[filtered_df['is_churned'] == 1]['monetary'].sum():,.2f}")
        
        html_summary = f"""
        <html>
        <head><title>Executive Churn Summary</title></head>
        <body>
            <h1>E-Commerce Retention & Churn Executive Summary</h1>
            <p><b>Total Customers:</b> {len(filtered_df)}</p>
            <p><b>Churn Rate:</b> {round((filtered_df['is_churned'].sum() / max(len(filtered_df), 1)) * 100, 2)}%</p>
            <p><b>At Risk Revenue:</b> ₹{filtered_df[filtered_df['is_churned'] == 1]['monetary'].sum():,.2f}</p>
        </body>
        </html>
        """
        
        st.download_button(
            label="Download Executive Summary (HTML)",
            data=html_summary,
            file_name="executive_summary.html",
            mime="text/html"
        )

# --- PAGE 3: MODEL EVALUATION & METRICS ---
elif page == "📊 Model Evaluation & Metrics":
    st.header("Model Evaluation & Metrics")
    if 'y_test' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), x=['Active', 'Churned'], y=['Active', 'Churned'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col2:
            st.subheader("ROC-AUC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Baseline'))
            fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)

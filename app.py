import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Enterprise Churn & Revenue Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CLAUDE-STYLE DARK SIDEBAR & HIGH-CONTRAST THEME
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Dark background for sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #282828 !important;
    }
    
    /* Force crisp white text across all sidebar labels, radio titles, and options */
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Specific radio button text contrast fix */
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label span,
    label[data-baseweb="radio"] * {
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Action Buttons in Sidebar */
    .stButton > button {
        background-color: #212121 !important;
        color: #E0E0E0 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        text-align: left !important;
        padding: 8px 12px !important;
    }
    .stButton > button:hover {
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
    }
    
    /* Section Headers */
    .sidebar-section-header {
        color: #AAAAAA !important;
        font-size: 11px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Profile Box */
    .user-profile-box {
        margin-top: 25px;
        padding: 12px 14px;
        background-color: #1A1A1A;
        border: 1px solid #282828;
        border-radius: 8px;
        color: #E0E0E0 !important;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & FILE UPLOAD
# ---------------------------------------------------------
with st.sidebar:
    st.button("➕ New Analytics Session", use_container_width=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.button("📁 Projects", use_container_width=True)
    with col_btn2:
        st.button("📦 Artifacts", use_container_width=True)
        
    st.markdown('<div class="sidebar-section-header">Data Source Ingestion</div>', unsafe_allow_html=True)
    
    uploaded_cust = st.file_uploader("Upload Customers CSV", type=["csv"])
    uploaded_trans = st.file_uploader("Upload Transactions CSV", type=["csv"])
    
    st.markdown('<div class="sidebar-section-header">Analytics Views</div>', unsafe_allow_html=True)
    
    nav_selection = st.radio(
        label="Select View",
        label_visibility="collapsed",
        options=[
            "📌 Executive Control Panel",
            "📈 Advanced Curves & Analytics",
            "🤖 ML Model Insights & Simulation",
            "📊 Multi-Model Comparison & Metrics",
            "📋 Exportable Cohorts Data"
        ]
    )
    
    st.markdown('<div class="sidebar-section-header">Industrial Engine</div>', unsafe_allow_html=True)
    st.caption("• SQLite Analytics Core")
    st.caption("• Multi-Model Ensembling")
    st.caption("• Live Risk Scoring Active")
    
    st.markdown("""
        <div class="user-profile-box">
            <span>👤 <b>ABHISHEK AHIRE</b> • Data Engineer</span>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA PIPELINE & MULTI-MODEL ENGINE
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data(cust_file, trans_file):
    if cust_file is not None and trans_file is not None:
        customers = pd.read_csv(cust_file)
        transactions = pd.read_csv(trans_file)
    else:
        customers = pd.read_csv("customers.csv")
        transactions = pd.read_csv("transactions.csv")
    
    conn = sqlite3.connect(":memory:")
    customers.to_sql("customers", conn, index=False)
    transactions.to_sql("transactions", conn, index=False)
    
    rfm_df = pd.read_sql("""
        SELECT 
            c.customer_id,
            c.age,
            c.city_tier,
            c.signup_channel,
            COUNT(t.order_id) AS frequency,
            COALESCE(SUM(t.order_amount_inr), 0) AS monetary,
            CAST((JULIANDAY('2026-09-01') - JULIANDAY(MAX(t.order_date))) AS INT) AS recency
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id
    """, conn)
    
    rfm_df['recency'] = rfm_df['recency'].fillna(365)
    rfm_df['is_churned'] = (rfm_df['recency'] > 90).astype(int)
    
    rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    def assign_segment(row):
        score = row['R_Score'] + row['F_Score'] + row['M_Score']
        if score >= 12:
            return "Champions"
        elif score >= 9:
            return "Potential Loyalist"
        elif score >= 6:
            return "At Risk"
        else:
            return "Hibernating"
            
    rfm_df['Customer_Segment'] = rfm_df.apply(assign_segment, axis=1)
    return rfm_df

rfm_df = load_and_process_data(uploaded_cust, uploaded_trans)

# ML Training Pipeline for Multiple Models
X = rfm_df[['age', 'frequency', 'monetary', 'recency']]
y = rfm_df['is_churned']

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

model_results = {}
for name, m in models.items():
    m.fit(X, y)
    probs = m.predict_proba(X)[:, 1]
    preds = m.predict(X)
    model_results[name] = {
        "model": m,
        "probs": probs,
        "preds": preds,
        "acc": accuracy_score(y, preds),
        "prec": precision_score(y, preds, zero_division=0),
        "rec": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    }

# Default primary model probability for scoring
rfm_df['churn_prob'] = model_results["Random Forest"]["probs"]

# ---------------------------------------------------------
# PAGE ROUTING
# ---------------------------------------------------------
st.title("🛡️ Enterprise Churn & Revenue Intelligence Platform")

if nav_selection == "📌 Executive Control Panel":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Base", f"{len(rfm_df):,}")
    col2.metric("Base Churn Rate", f"{(rfm_df['is_churned'].mean() * 100):.1f}%")
    col3.metric("Avg Customer Lifetime Spend", f"₹{rfm_df['monetary'].mean():,.2f}")
    col4.metric("At-Risk Revenue Value", f"₹{rfm_df[rfm_df['is_churned'] == 1]['monetary'].sum():,.2f}")
    
    st.markdown("---")
    
    with st.expander("🤖 **Generate AI Executive Strategy Briefing**", expanded=True):
        highest_churn_channel = rfm_df.groupby('signup_channel')['is_churned'].mean().idxmax()
        at_risk_rev = rfm_df[rfm_df['is_churned'] == 1]['monetary'].sum()
        champions_pct = (rfm_df['Customer_Segment'] == 'Champions').mean() * 100
        
        st.markdown(f"""
        * **Revenue Alert**: **₹{at_risk_rev:,.2f}** is currently at high risk of churn across inactive cohorts.
        * **Acquisition Efficiency**: **{highest_churn_channel}** exhibits the highest churn rate. Recommend reallocating marketing spend to organic/referral channels.
        * **Retention Core**: **{champions_pct:.1f}%** of the base are Champions. Implementing a loyalty initiative for this segment will protect key recurring revenue.
        """)
        
        html_report = f"""
        <html>
        <head><title>Executive Churn Intelligence Brief</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2>Enterprise Churn & Revenue Intelligence Summary</h2>
            <p><b>Date:</b> September 2026</p>
            <hr>
            <p><b>Total Customers:</b> {len(rfm_df)}</p>
            <p><b>Churn Rate:</b> {(rfm_df['is_churned'].mean() * 100):.1f}%</p>
            <p><b>At-Risk Revenue:</b> ₹{at_risk_rev:,.2f}</p>
            <p><b>Highest Risk Channel:</b> {highest_churn_channel}</p>
        </body>
        </html>
        """
        st.download_button(
            label="📥 Download Executive Summary (HTML)",
            data=html_report,
            file_name="Executive_Churn_Summary.html",
            mime="text/html"
        )
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Customer Segment Allocation")
        fig1 = px.pie(rfm_df, names='Customer_Segment', hole=0.45, color_discrete_sequence=px.colors.sequential.Darkmint)
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.subheader("2. Signup Channel Churn Rates")
        chan_df = rfm_df.groupby('signup_channel')['is_churned'].mean().reset_index()
        chan_df['is_churned'] *= 100
        fig2 = px.bar(chan_df, x='signup_channel', y='is_churned', color='is_churned', color_continuous_scale='Reds', labels={'is_churned': 'Churn %'})
        st.plotly_chart(fig2, use_container_width=True)

elif nav_selection == "📈 Advanced Curves & Analytics":
    st.subheader("Industrial Visualization Suite (6 Advanced Analytics Curves)")
    
    col_f1, col_f2 = st.columns(2)
    selected_tier = col_f1.multiselect("Filter by City Tier", options=rfm_df['city_tier'].unique(), default=rfm_df['city_tier'].unique())
    selected_channel = col_f2.multiselect("Filter by Signup Channel", options=rfm_df['signup_channel'].unique(), default=rfm_df['signup_channel'].unique())
    
    filtered_df = rfm_df[
        (rfm_df['city_tier'].isin(selected_tier)) & 
        (rfm_df['signup_channel'].isin(selected_channel))
    ]
    
    if filtered_df.empty:
        st.warning("No data matching the selected filters.")
    else:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("### Curve 1: Recency vs Monetary Trajectory")
            fig_c1 = px.scatter(
                filtered_df, x='recency', y='monetary', color='Customer_Segment', size='frequency',
                labels={'recency': 'Days Since Last Purchase', 'monetary': 'Spend Value (₹)'},
                template="plotly_dark"
            )
            st.plotly_chart(fig_c1, use_container_width=True)
            
            st.markdown("### Curve 2: Pareto Cumulative Revenue Concentration Curve")
            pareto_df = filtered_df.sort_values(by='monetary', ascending=False).reset_index(drop=True)
            pareto_df['cum_revenue'] = pareto_df['monetary'].cumsum() / pareto_df['monetary'].sum() * 100
            pareto_df['cum_customers'] = (pareto_df.index + 1) / len(pareto_df) * 100
            
            fig_c2 = px.line(
                pareto_df, x='cum_customers', y='cum_revenue',
                labels={'cum_customers': '% of Total Customers', 'cum_revenue': 'Cumulative Revenue %'},
                template="plotly_dark"
            )
            fig_c2.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="Gray", dash="dash"))
            fig_c2.add_shape(type="line", x0=20, y0=0, x1=20, y1=100, line=dict(color="Red", dash="dot"))
            st.plotly_chart(fig_c2, use_container_width=True)

            st.markdown("### Curve 3: Recency vs Frequency Risk Heatmap")
            heatmap_data = filtered_df.pivot_table(index='R_Score', columns='F_Score', values='churn_prob', aggfunc='mean')
            fig_c3 = px.imshow(
                heatmap_data, labels=dict(x="Frequency Score", y="Recency Score", color="Avg Churn Risk"),
                color_continuous_scale="YlOrRd", template="plotly_dark"
            )
            st.plotly_chart(fig_c3, use_container_width=True)

        with col_c2:
            st.markdown("### Curve 4: Churn Probability Distribution Density Curve")
            fig_c4 = px.histogram(
                filtered_df, x='churn_prob', color='is_churned', nbins=30, marginal="box",
                labels={'churn_prob': 'Predicted Churn Probability'},
                color_discrete_map={0: "#00CC96", 1: "#EF553B"}, template="plotly_dark"
            )
            st.plotly_chart(fig_c4, use_container_width=True)

            st.markdown("### Curve 5: Age Bracket Churn Susceptibility Curve")
            age_bins = pd.cut(filtered_df['age'], bins=[18, 25, 35, 50, 65, 80])
            age_churn = filtered_df.groupby(age_bins, observed=False)['is_churned'].mean().reset_index()
            age_churn['age'] = age_churn['age'].astype(str)
            age_churn['is_churned'] *= 100
            
            fig_c5 = px.line(
                age_churn, x='age', y='is_churned', markers=True,
                labels={'age': 'Age Group', 'is_churned': 'Churn Risk %'}, template="plotly_dark"
            )
            st.plotly_chart(fig_c5, use_container_width=True)

            st.markdown("### Curve 6: Random Forest Feature Importance Rankings")
            importances = model_results["Random Forest"]["model"].feature_importances_
            features = ['Age', 'Frequency', 'Monetary Value', 'Recency']
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
            
            fig_c6 = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                labels={'Importance': 'Relative Importance Weight'}, color='Importance', template="plotly_dark"
            )
            st.plotly_chart(fig_c6, use_container_width=True)

elif nav_selection == "🤖 ML Model Insights & Simulation":
    st.subheader("Real-Time Churn Risk Simulator Engine")
    
    selected_sim_model = st.selectbox("Select Prediction Model Engine", options=["Random Forest", "XGBoost", "Logistic Regression"])
    active_m = model_results[selected_sim_model]["model"]
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    age_in = col_s1.slider("Age", 18, 80, 28)
    freq_in = col_s2.slider("Orders Count", 1, 30, 2)
    mon_in = col_s3.slider("Total Spend (₹)", 500, 50000, 3500)
    rec_in = col_s4.slider("Recency (Days)", 1, 365, 120)
    
    pred_prob = active_m.predict_proba([[age_in, freq_in, mon_in, rec_in]])[0][1]
    
    st.markdown("---")
    st.markdown(f"### Predicted Churn Risk ({selected_sim_model}): **{pred_prob * 100:.1f}%**")
    st.progress(float(pred_prob))
    
    if pred_prob >= 0.6:
        st.error("🚨 **High Risk Warning**: Customer requires immediate engagement or promotional incentives.")
    else:
        st.success("✅ **Healthy Customer**: High retention likelihood.")

elif nav_selection == "📊 Multi-Model Comparison & Metrics":
    st.subheader("Multi-Model Validation & Performance Diagnostics")
    
    # Leaderboard Summary Table
    metrics_summary = []
    for m_name, res in model_results.items():
        metrics_summary.append({
            "Model Engine": m_name,
            "Accuracy": f"{res['acc'] * 100:.2f}%",
            "Precision": f"{res['prec'] * 100:.2f}%",
            "Recall": f"{res['rec'] * 100:.2f}%",
            "F1-Score": f"{res['f1'] * 100:.2f}%"
        })
    st.markdown("### Model Benchmark Leaderboard")
    st.table(pd.DataFrame(metrics_summary))
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("### Combined ROC-AUC Curves")
        fig_roc = go.Figure()
        
        for m_name, res in model_results.items():
            fpr, tpr, _ = roc_curve(y, res['probs'])
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{m_name} (AUC = {roc_auc:.3f})', mode='lines'))
            
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Chance', mode='lines', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_m2:
        st.markdown("### Confusion Matrix Selector")
        chosen_cm_model = st.selectbox("View Confusion Matrix for:", options=["Random Forest", "XGBoost", "Logistic Regression"])
        cm = confusion_matrix(y, model_results[chosen_cm_model]["preds"])
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Label"),
            x=['Retained (0)', 'Churned (1)'],
            y=['Retained (0)', 'Churned (1)'],
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

elif nav_selection == "📋 Exportable Cohorts Data":
    st.subheader("Raw Customer Base & Risk Ratings")
    st.dataframe(rfm_df, use_container_width=True)
    
    csv_data = rfm_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export High-Risk Cohorts to CSV", csv_data, "churn_cohorts.csv", "text/csv")
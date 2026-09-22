import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(
    page_title="Enterprise Churn & Revenue Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CLAUDE-STYLE DARK SIDEBAR & INDUSTRIAL THEME
# ---------------------------------------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #282828 !important;
    }
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
    .sidebar-section-header {
        color: #888888;
        font-size: 11px;
        font-weight: 600;
        margin-top: 16px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .user-profile-box {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 21rem;
        padding: 12px 16px;
        background-color: #121212;
        border-top: 1px solid #282828;
        color: #E0E0E0;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.button("➕ New Analytics Session", use_container_width=True)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.button("📁 Projects", use_container_width=True)
    with col_btn2:
        st.button("📦 Artifacts", use_container_width=True)
        
    st.markdown('<div class="sidebar-section-header">Analytics Views</div>', unsafe_allow_html=True)
    
    nav_selection = st.radio(
        label="Navigation",
        options=[
            "📌 Executive Control Panel",
            "📈 Advanced Curves & Analytics",
            "🔮 ML Model Insights & Simulation",
            "📊 Model Evaluation & Metrics",
            "📋 Exportable Cohorts Data"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="sidebar-section-header">Industrial Engine</div>', unsafe_allow_html=True)
    st.caption("• SQLite Analytics Core")
    st.caption("• Random Forest Classifier v1.4")
    st.caption("• Live Risk Scoring Active")
    
    st.markdown("""
        <div class="user-profile-box">
            <span>👤 <b>ABHISHEK AHIRE</b> • Data Engineer</span>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA PIPELINE & ML ENGINE
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
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

rfm_df = load_and_process_data()

# ML Model Training
X = rfm_df[['age', 'frequency', 'monetary', 'recency']]
y = rfm_df['is_churned']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
rfm_df['churn_prob'] = model.predict_proba(X)[:, 1]

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
    
    # FEATURE 1: AUTOMATED AI INSIGHT ENGINE
    with st.expander("🤖 **Generate AI Executive Strategy Briefing**", expanded=True):
        highest_churn_channel = rfm_df.groupby('signup_channel')['is_churned'].mean().idxmax()
        at_risk_rev = rfm_df[rfm_df['is_churned'] == 1]['monetary'].sum()
        champions_pct = (rfm_df['Customer_Segment'] == 'Champions').mean() * 100
        
        st.markdown(f"""
        * **Revenue Alert**: **₹{at_risk_rev:,.2f}** is currently at high risk of churn across inactive cohorts[cite: 6].
        * **Acquisition Efficiency**: **{highest_churn_channel}** exhibits the highest churn rate[cite: 6]. Recommend reallocating marketing spend to organic/referral channels.
        * **Retention Core**: **{champions_pct:.1f}%** of the base are Champions. Implementing a loyalty initiative for this segment will protect key recurring revenue.
        """)
        
        # FEATURE 4: HTML EXECUTIVE REPORT GENERATOR
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
    
    # FEATURE 2: GLOBAL FILTER BAR ACROSS ALL CURVES
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
            # CURVE 1
            st.markdown("### Curve 1: Recency vs Monetary Trajectory")
            fig_c1 = px.scatter(
                filtered_df, x='recency', y='monetary', color='Customer_Segment', size='frequency',
                labels={'recency': 'Days Since Last Purchase', 'monetary': 'Spend Value (₹)'},
                template="plotly_dark"
            )
            st.plotly_chart(fig_c1, use_container_width=True)
            
            # CURVE 2
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

            # CURVE 3
            st.markdown("### Curve 3: Recency vs Frequency Risk Heatmap")
            heatmap_data = filtered_df.pivot_table(index='R_Score', columns='F_Score', values='churn_prob', aggfunc='mean')
            fig_c3 = px.imshow(
                heatmap_data, labels=dict(x="Frequency Score", y="Recency Score", color="Avg Churn Risk"),
                color_continuous_scale="YlOrRd", template="plotly_dark"
            )
            st.plotly_chart(fig_c3, use_container_width=True)

        with col_c2:
            # CURVE 4
            st.markdown("### Curve 4: Churn Probability Distribution Density Curve")
            fig_c4 = px.histogram(
                filtered_df, x='churn_prob', color='is_churned', nbins=30, marginal="box",
                labels={'churn_prob': 'Predicted Churn Probability'},
                color_discrete_map={0: "#00CC96", 1: "#EF553B"}, template="plotly_dark"
            )
            st.plotly_chart(fig_c4, use_container_width=True)

            # CURVE 5
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

            # CURVE 6
            st.markdown("### Curve 6: ML Feature Importance Driver Rankings")
            importances = model.feature_importances_
            features = ['Age', 'Frequency', 'Monetary Value', 'Recency']
            fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
            
            fig_c6 = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                labels={'Importance': 'Relative Importance Weight'}, color='Importance', template="plotly_dark"
            )
            st.plotly_chart(fig_c6, use_container_width=True)

elif nav_selection == "🔮 ML Model Insights & Simulation":
    st.subheader("Real-Time Churn Risk Simulator Engine")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    age_in = col_s1.slider("Age", 18, 80, 28)
    freq_in = col_s2.slider("Orders Count", 1, 30, 2)
    mon_in = col_s3.slider("Total Spend (₹)", 500, 50000, 3500)
    rec_in = col_s4.slider("Recency (Days)", 1, 365, 120)
    
    pred_prob = model.predict_proba([[age_in, freq_in, mon_in, rec_in]])[0][1]
    
    st.markdown("---")
    st.markdown(f"### Predicted Churn Risk Index: **{pred_prob * 100:.1f}%**")
    st.progress(float(pred_prob))
    
    if pred_prob >= 0.6:
        st.error("🚨 **High Risk Warning**: Customer requires immediate engagement or promotional incentives.")
    else:
        st.success("✅ **Healthy Customer**: High retention likelihood.")

# FEATURE 3: MODEL EVALUATION & DIAGNOSTICS TAB
elif nav_selection == "📊 Model Evaluation & Metrics":
    st.subheader("Random Forest Model Validation & Performance Diagnostics")
    
    y_pred = model.predict(X)
    y_prob = rfm_df['churn_prob']
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted Label", y="Actual Label"),
            x=['Retained (0)', 'Churned (1)'],
            y=['Retained (0)', 'Churned (1)'],
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_m2:
        st.markdown("### ROC-AUC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Random Forest (AUC = {roc_auc:.3f})', mode='lines', line=dict(color='#00CC96', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Chance', mode='lines', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
        st.plotly_chart(fig_roc, use_container_width=True)

elif nav_selection == "📋 Exportable Cohorts Data":
    st.subheader("Raw Customer Base & Risk Ratings")
    st.dataframe(rfm_df, use_container_width=True)
    
    csv_data = rfm_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export High-Risk Cohorts to CSV", csv_data, "churn_cohorts.csv", "text/csv")
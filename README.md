# 🛒 E-Commerce Churn & Revenue Intelligence Engine

> **A production-ready data pipeline and ML analytics platform designed to pinpoint high-risk customer cohorts, model lifetime spending patterns, and prevent revenue leakage.**

---

## ⚡ The Big Picture

Retaining existing customers is significantly cheaper than acquiring new ones, yet identifying *who* is about to walk away remains a challenge.

This project builds an end-to-end intelligence hub that ingests raw transaction feeds, processes RFM metrics via SQLite, and deploys multi-model Machine Learning (**Random Forest**, **XGBoost**, and **Logistic Regression**) to forecast churn risk in real time.

---

## 🚀 Key Features & Capabilities

* **Dynamic Data Ingestion**: Supports real-time CSV uploads or seamlessly falls back to synthetic data pipelines.
* **In-Memory SQLite Engine**: High-performance SQL queries aggregate Recency, Frequency, and Monetary (RFM) scores directly on the fly.
* **Multi-Model Benchmark Suite**: Compares **Random Forest**, **XGBoost**, and **Logistic Regression** across ROC-AUC curves, confusion matrices, and F1-scores.
* **Real-Time Risk Simulator**: Adjust live customer attributes (age, purchase frequency, recency, order values) to evaluate churn probabilities on demand.
* **Executive Summary Generator**: Automates downloading high-level strategic briefs formatted in clean HTML for leadership handoffs.

---

## 🛠️ Tech Stack & Architecture

| Layer | Tech Stack |
| :--- | :--- |
| **Interface / Dashboard** | Streamlit |
| **Data Processing** | Python, Pandas, NumPy |
| **Database Engine** | SQLite3 |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Visualizations** | Plotly Express, Plotly Graph Objects |

---

## 🚦 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY

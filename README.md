# Credit Card Fraud Detection – End-to-End Data Science Project
🚨 Detecting fraudulent transactions using machine learning and explainable AI (SHAP) 🚨

# Project Overview
This project builds and evaluates machine learning models to detect fraudulent credit card transactions.
The dataset is highly imbalanced, with less than 1% of transactions marked as fraud, making this a challenging classification task.

# Key objectives:
- Perform EDA to understand fraud patterns
- Engineer meaningful features from raw data
- Compare multiple models (Logistic Regression, Random Forest, XGBoost)
- Apply SHAP to explain model predictions
- Provide actionable insights for fraud prevention teams

# Dataset
- Rows: 339,607
- Fraud rate: 0.525%
- Features include transaction amount, category, location, time, and customer demographics.
- Dataset used for educational purposes.

# EDA Highlights
- Fraud concentrated in certain amount ranges (500–2.5k USD)
- Higher fraud risk at night hours (0–3am, 10–11pm) and Fridays
- Certain merchant categories (e.g., online shopping) have higher fraud rates
- Higher fraud rates in specific states (AK, OR, NE, CO, NM)
- Younger (18–24) and older (55–64) age groups more prone to fraud

# Feature Engineering
Created new features to improve model performance:
- amt_bucket – Transaction amount range
- is_high_risk_category – Flag for high-risk merchant categories
- age_bucket – Customer age group
- trans_hour – Hour of transaction
- trans_dow – Day of week
- is_night – Night-time transaction flag
- state_fraud_rate – Fraud rate per state
- merchant_distance_km – Distance between customer and merchant

# Modeling
Trained and compared:
1. Logistic Regression
2.Random Forest
3.XGBoost

Evaluation Metrics:
1.ROC-AUC
2. PR-AUC
3. Precision, Recall, F1-score
4. Confusion Matrix

# Results
| Model               | Recall | Precision | F1-score | ROC-AUC | PR-AUC |
| ------------------- | ------ | --------- | -------- | ------- | ------ |
| Logistic Regression | 0.8961 | 0.0538    | 0.1016   | 0.9640  | 0.2478 |
| Random Forest       | 0.4916 | 0.8929    | 0.6341   | 0.9648  | 0.6934 |
| XGBoost             | 0.8230 | 0.2954    | 0.4347   | 0.9860  | 0.7377 |


# Explainable AI (SHAP)
- Top features impacting predictions:
  1. amt
  2. is_night
  3.age_years
  4. trans_hour
  5. is_high_risk_category
  6. merchant_distance_km
  7. state_fraud_rate
- SHAP visualizations used for both global feature importance and local predictions.

# Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook

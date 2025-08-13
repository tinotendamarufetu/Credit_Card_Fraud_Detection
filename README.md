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

# Key Insights from EDA
1. Class Imbalance
- Only 1,782 fraudulent transactions vs 337,825 non-fraudulent transactions.
- Special handling required to avoid bias toward non-fraud class.

2. Amount Analysis
- Fraud often involves higher transaction amounts (especially 500–2500 range).
- Fraud rate peaks at 23.7% for transactions between 1k–2.5k.

3. Temporal Patterns
- Fraud spikes during night hours (9 PM–3 AM).
- Certain weekdays (Thursday & Friday) show slightly higher fraud rates.

4. Geographic Risk
- Certain states (e.g., UT, ID, AZ) have noticeably higher fraud percentages.
- Merchant and customer distance sometimes correlates with fraud.

5. High-Risk Categories
- Categories like shopping_net, grocery_pos, and misc_pos show higher fraud tendencies.

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
- Most influential feature: Transaction amount (amt)
- Night transactions (is_night) and age are also strong fraud indicators.
- SHAP helped visualize how individual features push predictions toward fraud or non-fraud.

# Key Takeaways
- Fraud detection is highly imbalanced — precision/recall trade-off is crucial.
- Amount, night-time activity, and geographic location are key signals.
- SHAP is invaluable for building trust and interpretability in fraud models.

# Visual Highlights
<img width="944" height="310" alt="image" src="https://github.com/user-attachments/assets/814c5e7d-1810-450f-a16d-2b0035290c54" />
<img width="627" height="309" alt="image" src="https://github.com/user-attachments/assets/3e51ad25-5875-469b-8df9-9c610db1141d" />
<img width="1481" height="467" alt="image" src="https://github.com/user-attachments/assets/7f92b8bd-7194-448b-8f74-742a3f84d058" />
<img width="1429" height="308" alt="image" src="https://github.com/user-attachments/assets/a26c294d-b74f-4dbb-a941-4e7693dc0a39" />




# Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook

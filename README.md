# AI Customer Churn Prediction & Decision Intelligence System

## Overview

An AI-powered customer churn prediction system that predicts customers who are likely to churn, explains key risk drivers, segments customers by risk level, estimates revenue exposure, and provides retention recommendations.

## Why I Built It

The goal was not just to predict churn, but to make the prediction useful for business decision-making.

The system is designed to answer:
- Which customers are most at risk?
- Why are they at risk?
- How much revenue could be affected?
- Which customers should be prioritized?

## Key Decisions

### Model Selection

I trained and compared Logistic Regression and Random Forest rather than selecting a model upfront.

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8427 |
| Random Forest | 0.8264 |

Logistic Regression was selected as the final model based on its higher ROC-AUC score.

I used ROC-AUC rather than relying only on accuracy because the churn dataset is imbalanced.

### Explainable Predictions

I analyzed feature importance to understand the factors driving churn rather than treating the model as a black box.

The strongest factors included:
- Total Charges
- Tenure
- Monthly Charges

These insights were incorporated into the dashboard to make the predictions easier to understand and act upon.

### Risk Segmentation

I defined Low, Medium, and High risk thresholds to make predictions easier to prioritize.

### Revenue Risk

The system includes a revenue-at-risk estimate to connect churn predictions with a business outcome, identifying approximately **$10.5K/month in potential revenue exposure** based on the dataset.

## Features

- Customer churn prediction
- Explainable AI insights
- Low/Medium/High risk segmentation
- Revenue-at-risk estimation
- Retention recommendations
- Interactive Streamlit dashboard
- Model comparison and evaluation

## Tech Stack

Python, Pandas, Scikit-learn, Streamlit

## Dashboard Preview

### Churn Prediction
<img width="959" height="497" alt="Screenshot 2026-08-11 150220" src="https://github.com/user-attachments/assets/669dc16c-3971-434e-a80d-cc3aaf721358" />


### Explainable AI
<img width="879" height="313" alt="Screenshot 2026-08-11 150252" src="https://github.com/user-attachments/assets/e9ec22a0-a2fb-4759-9d5c-e67f39fb47a1" />


### Retention Recommendations
<img width="854" height="242" alt="Screenshot 2026-08-11 150304" src="https://github.com/user-attachments/assets/3e93cea2-26ac-4297-ac56-5e496bbfde9a" />


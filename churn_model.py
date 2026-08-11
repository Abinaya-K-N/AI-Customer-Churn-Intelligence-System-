# ============================================
# AI Customer Churn Intelligence System
# Model Training Pipeline
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import warnings

warnings.filterwarnings("ignore")

print("\nLoading dataset...")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data/customer_churnn.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape, "\n")

# ---------------------------
# Data Cleaning
# ---------------------------

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Remove ID column
df.drop('customerID', axis=1, inplace=True)

# ---------------------------
# Encoding categorical features
# ---------------------------
df = pd.get_dummies(df, drop_first=True)

# Ensure no missing values remain
df.fillna(0, inplace=True)

print("Data Cleaning Completed\n")

# ---------------------------
# Train-Test Split
# ---------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Completed\n")

# =====================================================
# Logistic Regression (FINAL DEPLOYED MODEL)
# =====================================================
print("Training Logistic Regression model...")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_prob_log = log_model.predict_proba(X_test)[:, 1]
log_auc = roc_auc_score(y_test, y_prob_log)

print("Logistic Regression ROC-AUC:", round(log_auc, 4))

# ---------------------------
# Save Model for Dashboard
# ---------------------------
with open("logistic_churn_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

print("Model saved successfully as logistic_churn_model.pkl\n")

# =====================================================
# Random Forest (Comparison Model)
# =====================================================
print("Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, y_prob_rf)

print("Random Forest ROC-AUC:", round(rf_auc, 4))

# ---------------------------
# Feature Importance
# ---------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# ---------------------------
# Risk Segmentation
# ---------------------------
risk_df = X_test.copy()
risk_df["Churn_Probability"] = y_prob_rf

def risk_category(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df["Risk_Level"] = risk_df["Churn_Probability"].apply(risk_category)

print("\nRisk Distribution:")
print(risk_df["Risk_Level"].value_counts())

# ---------------------------
# Revenue at Risk Calculation
# ---------------------------
risk_df["MonthlyCharges"] = X_test["MonthlyCharges"]

high_risk = risk_df[risk_df["Risk_Level"] == "High Risk"]

revenue_at_risk = high_risk["MonthlyCharges"].sum()

print("\nEstimated Monthly Revenue at Risk: $",
      round(revenue_at_risk, 2))

print("\nProject Execution Completed Successfully.")
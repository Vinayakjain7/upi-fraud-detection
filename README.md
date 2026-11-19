# UPI Fraud Detection using XGBoost and SMOTE

ML course project (Machine Learning & Stochastic Processes, Bennett University) pitched as a startup: **PayShield**, an API that scores every UPI-style transaction for fraud.

**Team:** Vinayak Jain (A25ARIU0013), Kartik (A25ARIU0019), Keshav (A25ARIU0020)

## Problem
Fraud is under 1% of transactions, so a model that always says "genuine" looks 99.7% accurate but catches nothing. We balance the data with SMOTE and judge models on recall and F1.

## Data
[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1): 6.3 million simulated mobile-money transactions. We use a random 10% sample and keep only TRANSFER and CASH_OUT (the only types with fraud).

## Method
1. Encode transaction type, drop ID columns
2. Add two balance-error features (our addition):
   - `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`
   - `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`
3. 80/20 stratified train-test split
4. SMOTE on training data only
5. Compare Logistic Regression, Random Forest and XGBoost

## Results (test set: 55,427 transactions, 163 frauds)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression + SMOTE | 0.057 | 0.933 | 0.107 |
| Random Forest + SMOTE | 1.000 | 1.000 | 1.000 |
| XGBoost (no SMOTE) | 0.982 | 1.000 | 0.991 |
| **XGBoost + SMOTE** | **0.988** | **1.000** | **0.994** |
| XGBoost + SMOTE, without our features | 0.422 | 0.975 | 0.589 |

XGBoost + SMOTE caught all 163 frauds with 2 false alarms. Our balance-error features raised F1 from 0.59 to 0.99.

**Limitation:** PaySim is simulated, so fraud is easier to spot than in real UPI data.

## How to run
Open `upi_fraud_detection.ipynb` in Google Colab and click Runtime → Run all. The dataset downloads automatically.

## Credits
- Base paper: Wenmeng Li, *Credit Card Fraud Detection: A System Based on Imbalanced Learning and Ensemble Models*
- Base code: [Jabezng2/Credit-Card-Fraud-Detection](https://github.com/Jabezng2/Credit-Card-Fraud-Detection)

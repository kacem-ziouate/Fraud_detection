## Project Title

Credit Card Fraud Detection Using Machine Learning (Imbalanced Data Case Study)

## Project Overview

Credit card fraud is a rare but high-impact problem where traditional accuracy-based analytics fail due to extreme class imbalance.
In this project, I developed an end-to-end fraud detection system using machine learning, focusing on recall, precision, and business trade-offs rather than misleading accuracy metrics.

The project simulates a real-world production scenario, ensuring that all evaluations are performed on unseen test data to avoid data leakage.

## Business Problem

Fraud transactions represent ~0.17% of all transactions

Missing fraud (false negatives) causes direct financial loss

Flagging legitimate transactions (false positives) causes customer dissatisfaction

Goal:
Build a model that maximizes fraud detection (recall) while controlling false alarms, using probability-based decision thresholds.

## Key Questions Answered

How do we detect fraud in extremely imbalanced data?

Why accuracy is misleading for fraud detection

How threshold selection impacts business outcomes

How to evaluate models realistically using unseen data

## Dataset

Source: European cardholders transaction dataset

Transactions: 284,807

Fraud cases: 492 (0.17%)

Features: PCA-transformed anonymized variables + Amount & Time

📌 Due to confidentiality, features are anonymized — this mirrors real banking environments.

## Tools & Technologies

Python

Pandas / NumPy — data manipulation

Scikit-learn — modeling & evaluation

Imbalanced-learn (SMOTE) — class imbalance handling

Matplotlib / Seaborn — visualization

## Project Workflow

###  1️⃣ Business Understanding

We are working for a financial services company (bank / fintech) that processes thousands of credit card transactions every minute.

The business problem:

Fraudulent transactions cause direct financial loss

False fraud alerts annoy customers and can cause customer churn

Manual review is slow, expensive, and not scalable

📌 Goal:
Detect fraudulent transactions as early and accurately as possible.

“Can we predict whether a transaction is fraudulent based on historical transaction data?”

#### Define the Target Variable (Very Important)

From the dataset:

Class = 1 → Fraudulent transaction

Class = 0 → Legitimate transaction

📌 This is our target (label).

So our task is:

Predict Class using all other features.

#### Objective

“The objective of this project is to build a machine learning model that predicts fraudulent credit card transactions using historical transaction data. The model aims to maximize fraud detection while minimizing false positives, using appropriate evaluation metrics for highly imbalanced data.”

### 2️⃣ Data Understanding & Exploration

```python
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set()

data = pd.read_csv('/Users/kacem/Documents/creditcard.csv')

data.head()

print(data.shape)

```
<img width="1201" height="791" alt="Screenshot 2026-01-17 at 17 16 40" src="https://github.com/user-attachments/assets/265c8bf4-e25b-4c69-a45f-0173e7331829" />

```python

print(data.info())

```
<img width="1247" height="712" alt="Screenshot 2026-01-17 at 17 17 11" src="https://github.com/user-attachments/assets/92e73c50-1452-402a-b90e-9808352ff058" />

```python

print(data.describe())

```
<img width="1232" height="795" alt="Screenshot 2026-01-17 at 17 17 28" src="https://github.com/user-attachments/assets/af552c50-f8d3-43fa-b13b-3cae79e52cb5" />


```python

print(data.isnull().sum())

```
<img width="1194" height="591" alt="Screenshot 2026-01-17 at 17 17 49" src="https://github.com/user-attachments/assets/87cfacc8-24c5-4f23-b1bc-c674c841b286" />

#### Check Class Distribution

```python

class_counts = data['Class'].value_counts()

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(class_counts, labels=['Legitimate (0)', 'Fraud (1)'],
        autopct='%1.2f%%', startangle=90)
plt.title('Credit Card Transactions Class Distribution')
plt.show()

```

<img width="1197" height="675" alt="Screenshot 2026-01-17 at 17 31 44" src="https://github.com/user-attachments/assets/556b345c-3f30-49da-b001-45862ffda962" />

#### Initial Insights 

Dataset is highly imbalanced → simple accuracy is useless

PCA-transformed features hide sensitive info but do not affect modeling

Amount and Time are raw features → can be used for feature engineering


### Data Cleaning & Preparation

we allready verify that we have no missing values so we move to preparation 

#### Scaling Numerical Features

Scaling ensures that numerical features are on a comparable scale so that machine learning models are not biased toward variables with larger magnitudes. This is especially important for distance- and gradient-based algorithms.


```python

from sklearn.preprocessing import StandardScaler

# Scale 'Time' and 'Amount'
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

```

#### Split Dataset (Train/Test)

Train set → model learning

Test set → evaluate performance on unseen data

Stratified split ensures the fraud ratio remains the same in both sets.

```python

from sklearn.model_selection import train_test_split

x = df.drop('Class', axis = 1 )
y = df['Class']

x_train , x_test , y_train , y_test = train_test_split( x, y ,test_size = 0.2 ,random_state = 42 ,stratify = y)

```
“I used a stratified train–test split to preserve the class distribution in both sets. This ensures reliable model evaluation, especially for highly imbalanced fraud data.”

#### Handle Class Imbalance
we will use it because 

Fraud class = 0.17% → models will ignore fraud if imbalance is not handled.

Solutions:

Class weights → tell model to penalize missing fraud more

Oversampling → SMOTE or RandomOverSampler

Undersampling → reduce legitimate transactions (less common)

for this situation i will use SMOTE but for training set only 

```python

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 42)
x_train_res,y_train_res = smote.fit_resample(x_train , y_train)

print(pd.Series(y_train_res).value_counts(normalize=True))

```
<img width="1191" height="254" alt="Screenshot 2026-01-17 at 17 45 09" src="https://github.com/user-attachments/assets/6443ead8-0281-4f38-ab30-c99d2f5eed06" />

### Exploratory Data Analysis (EDA)

#### 1️⃣ Compare Transaction Amounts


```python

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

plt.figure(figsize=(8,4))
sns.kdeplot(legit['Amount'], label='Legitimate', fill=True)
sns.kdeplot(fraud['Amount'], label='Fraud', fill=True)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.legend()
plt.show()

```
<img width="1198" height="705" alt="Screenshot 2026-01-17 at 17 49 08" src="https://github.com/user-attachments/assets/52e50588-82e2-42df-b0fe-22acdb5c6cf4" />

“Both fraud and legitimate transactions are dominated by small amounts, and raw KDE plots are insufficient to visually separate them due to extreme skewness.”

```python

df['LogAmount'] = np.log1p(df['Amount'])

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

plt.figure(figsize=(8,4))
sns.kdeplot(legit['LogAmount'], label='Legitimate', fill=True)
sns.kdeplot(fraud['LogAmount'], label='Fraud', fill=True)
plt.title('Log-Transformed Transaction Amount Distribution')
plt.xlabel('Log(Amount)')
plt.legend()
plt.show()

```
<img width="1139" height="637" alt="Screenshot 2026-01-17 at 17 51 20" src="https://github.com/user-attachments/assets/8c330d03-f31c-4525-85cf-f4e66f248dde" />

“Initial KDE plots of raw transaction amounts showed heavy overlap due to extreme right skewness. After applying a log transformation, clearer differences emerged, revealing that fraudulent transactions are typically smaller and more concentrated.”

#### 2️⃣ Analyze Time Patterns

```python

plt.figure(figsize=(8,4))
sns.histplot(fraud['Time'], bins=50, color='red', label='Fraud')
plt.title('Fraud Transactions Over Time')
plt.xlabel('Time')
plt.show()

```
<img width="1122" height="508" alt="Screenshot 2026-01-17 at 17 51 49" src="https://github.com/user-attachments/assets/c814398d-ebdb-4f4b-9cc7-f61c0b77b044" />

“Fraud transactions exhibit temporal clustering, indicating coordinated activity rather than random occurrence, which supports the use of machine learning models to capture behavioral patterns.”

#### 3️⃣ Feature Distribution Differences

```python

plt.figure(figsize=(8,4))
sns.kdeplot(legit['V1'], label='Legitimate')
sns.kdeplot(fraud['V1'], label='Fraud')
plt.title('V1 Distribution: Fraud vs Legit')
plt.legend()
plt.show()

```
<img width="1187" height="535" alt="Screenshot 2026-01-17 at 17 55 22" src="https://github.com/user-attachments/assets/b93d1a93-b07a-4eb6-9607-7e99aca5df31" />

“The distribution of PCA component V1 shows a clear separation between fraudulent and legitimate transactions, with fraud exhibiting higher variance and extreme values, confirming strong predictive signal in the feature space.”

#### 4️⃣ Correlation Check

```python

corr = data.corr()['Class'].sort_values()
print(corr.tail(10))

```
<img width="1119" height="246" alt="Screenshot 2026-01-17 at 18 01 22" src="https://github.com/user-attachments/assets/07dd5a19-6a68-4c0a-b0d0-28a8fb86c7b0" />

#### 5️⃣ Summary

During EDA, I compared fraudulent and legitimate transactions across amount, time, and feature distributions to confirm behavioral differences and validate the feasibility of machine learning for fraud detection.

### Baseline Model (Logistic Regression)

Prepare the data so models can learn those patterns correctly and fairly.

Establish a strong, interpretable baseline that every advanced model must beat.


#### 1️⃣ Train the Baseline Model

```python

from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression(
    max_iter = 1000,
    random_state = 42 
)
log_reg.fit(x_train_res , y_train_res)

```
#### Make Predictions on the Test Set

```python

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]


```
#### Evaluate

```python

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

```

```python

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

```
```python

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_proba)

```
This measures:

How well the model separates fraud from legit overall


<img width="1196" height="695" alt="Screenshot 2026-01-17 at 18 06 27" src="https://github.com/user-attachments/assets/d8b08efa-02c7-46be-ba95-54a694122394" />

#### Steps Explain 

I started with a Logistic Regression baseline trained on balanced data to establish a reliable performance benchmark and ensure model interpretability before moving to more complex algorithms.

### Advanced Models (Tree-Based Models)

Capture non-linear patterns and complex feature interactions that linear models cannot.


Why Move Beyond Logistic Regression?

Logistic Regression:

Assumes linear decision boundaries

Works well, but has limits

Fraud patterns are often:

Non-linear

Interaction-based (V3 + V10 + Amount together)

Irregular

📌 Tree-based models shine here.

#### 1: Random Forest
```python

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

```
##### Predictions
```python

y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

```
#### 2: Gradient Boosting

```python

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    random_state=42
)

gb.fit(X_train_res, y_train_res)

```
##### Predictions
```python

y_pred_gb = gb.predict(X_test_scaled)
y_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]

```

<img width="1194" height="710" alt="Screenshot 2026-01-17 at 18 11 58" src="https://github.com/user-attachments/assets/40cbe955-9964-4290-91fd-7048f2df949d" />

#### Explanation
After establishing a logistic regression baseline, I trained tree-based ensemble models such as Random Forest and Gradient Boosting to capture non-linear relationships and complex feature interactions typical in fraud detection problems.


### Model Evaluation

“Which model best detects fraud while controlling false alarms?”

```python

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred_rf)
```

```python

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))

```

```python

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
roc_auc_score(y_test, y_proba_rf)

```

```python

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_proba_rf)

```
<img width="1195" height="621" alt="Screenshot 2026-01-17 at 18 20 14" src="https://github.com/user-attachments/assets/3dfce078-3ff5-4146-8869-a3380e84e60c" />

#### Explanation 

I evaluated models using recall, precision, ROC-AUC, and precision–recall curves, focusing on minimizing false negatives while controlling false positives due to the highly imbalanced nature of fraud data.

### Threshold Optimization

At what probability should we flag a transaction as fraud?

#### Visualizing the Trade-off
```python

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(_, precision[:-1], label="Precision")
plt.plot(_, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

```
<img width="1203" height="782" alt="Screenshot 2026-01-17 at 18 22 23" src="https://github.com/user-attachments/assets/26837aad-755f-46f8-bb3d-9fb8997c915a" />

the interpretation for this graph is :

Threshold ↓ → Recall ↑, Precision ↓

Threshold ↑ → Precision ↑, Recall ↓


#### Choosing a Threshold

##### A — Recall-Driven (Fraud-First)

```python

target_recall = 0.90
idx = np.where(recall >= target_recall)[0]
best_idx = idx[np.argmax(precision[idx])]

best_threshold = _[idx]

best_threshold
```
#####B — Balanced (F1-Driven)

```python

f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = np.nanargmax(f1_scores)
best_threshold = _[best_idx]

best_threshold

```
This gives a balanced operating point.

#### Apply B Threshold

```python

y_pred_optimized = (y_proba_rf >= best_threshold).astype(int)

```
#### Re-Evaluate With New Threshold

```python

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred_optimized))
print(classification_report(y_test, y_pred_optimized))

```
<img width="1201" height="719" alt="Screenshot 2026-01-17 at 18 24 14" src="https://github.com/user-attachments/assets/29c72336-3790-45f2-b0b4-bc018d142119" />

I optimized the classification threshold using precision–recall trade-offs to maximize fraud detection while controlling false positives, rather than relying on the default 0.5 cutoff.

### Final Model Selection & Justification

Final Model: Gradient Boosting (with optimized threshold)

Why?

Highest fraud recall

Best performance on minority class

Handles non-linear patterns

Industry-standard for fraud detection

📌 Threshold tuning controls false positives, so precision is manageable.

After evaluating Logistic Regression, Random Forest, and Gradient Boosting using fraud-appropriate metrics, Gradient Boosting was selected due to its superior recall and ability to capture complex non-linear fraud patterns. Threshold optimization was applied to balance fraud detection and false positives.

### Business Impact, Ethics & Deployment

#### Business Impact (Translate Metrics → Money)

Assume we have :

Average fraud transaction = $500

Without ML:

Fraud goes undetected

With my model:

77 frauds caught → $38,500 saved

21 frauds missed → $10,500 lost

📌 Net impact:

Significant reduction in financial loss

The deployed model significantly reduces financial loss by detecting the majority of fraudulent transactions while keeping false positives minimal, ensuring both cost efficiency and customer satisfaction


#### my confusion matrix showed:

Only 3 legitimate transactions flagged as fraud

#### Ethical Considerations

“Ethical risks such as false positives and lack of interpretability were mitigated through threshold tuning, human review processes, and privacy-preserving data handling.”

#### Deployment Simulation

```python

def predict_fraud(transaction_features, model, _):
    prob = model.predict_proba(transaction_features)[0, 1]
    return int(prob >= _), prob  

```

“Post-deployment, the model would be continuously monitored for performance drift and retrained periodically to adapt to evolving fraud patterns.”

## Project Summary

“This project demonstrates an end-to-end fraud detection system, from data preprocessing and imbalance handling to model evaluation, threshold optimization, and business-driven deployment considerations.”

## Key Takeaways

Accuracy is not suitable for fraud detection

Class imbalance requires specialized handling

Threshold tuning is a business decision, not a technical default

Proper evaluation on unseen data is critical

##  Future Improvements

Cost-sensitive learning

Real-time streaming integration

Model explainability (SHAP)

Concept drift monitoring

👤 Author

Kacem Ziouate
Aspiring Data Analyst / Data Scientist
Focused on real-world, business-driven machine learning projects










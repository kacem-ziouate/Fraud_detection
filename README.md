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

###  Business Understanding

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

### Data Understanding & Exploration

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

















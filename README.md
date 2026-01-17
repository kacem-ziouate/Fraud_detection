📌 Project Title

Credit Card Fraud Detection Using Machine Learning (Imbalanced Data Case Study)

📖 Project Overview

Credit card fraud is a rare but high-impact problem where traditional accuracy-based analytics fail due to extreme class imbalance.
In this project, I developed an end-to-end fraud detection system using machine learning, focusing on recall, precision, and business trade-offs rather than misleading accuracy metrics.

The project simulates a real-world production scenario, ensuring that all evaluations are performed on unseen test data to avoid data leakage.

🎯 Business Problem

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

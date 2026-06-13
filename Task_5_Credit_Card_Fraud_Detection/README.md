# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

Credit card fraud is one of the most critical challenges faced by financial institutions worldwide. Fraudulent transactions result in significant financial losses and negatively impact customer trust.

The objective of this project is to develop a Machine Learning model capable of accurately identifying fraudulent credit card transactions while minimizing false alarms. Due to the highly imbalanced nature of the dataset, advanced resampling techniques and multiple classification algorithms were applied to improve fraud detection performance.

This project was completed as part of the **CodSoft Data Science Internship**.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions using Machine Learning.
* Handle highly imbalanced transaction data.
* Perform data preprocessing and feature scaling.
* Apply advanced resampling techniques.
* Train and compare multiple classification models.
* Evaluate model performance using fraud detection metrics.
* Identify the most effective model for fraud prediction.

---

## 📊 Dataset Information

The dataset contains anonymized credit card transactions made by European cardholders.

### Dataset Features

* **Time** – Time elapsed between transactions.
* **V1 – V28** – PCA-transformed numerical features.
* **Amount** – Transaction amount.
* **Class**

  * `0` → Genuine Transaction
  * `1` → Fraudulent Transaction

### Dataset Characteristics

* Highly imbalanced dataset.
* Fraud transactions represent a very small percentage of total transactions.
* Suitable for anomaly detection and classification problems.

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset overview and structure analysis.
* Missing value detection.
* Duplicate record analysis.
* Fraud vs Genuine transaction distribution.
* Transaction amount distribution.
* Correlation analysis.
* Correlation heatmap visualization.
* Class imbalance analysis.

---

## ⚙️ Data Preprocessing

To prepare the data for Machine Learning:

* Removed duplicate records.
* Standardized transaction amount and time features.
* Created scaled features:

  * Amount_Scaled
  * Time_Scaled
* Split data into training and testing datasets.
* Prepared features and target variables.

---

## ⚖️ Handling Class Imbalance

Since fraud detection datasets are extremely imbalanced, multiple resampling techniques were used:

### 1️⃣ SMOTE (Synthetic Minority Oversampling Technique)

* Generates synthetic fraud samples.
* Improves fraud class representation.

### 2️⃣ Random Under Sampling

* Reduces majority class samples.
* Creates a balanced dataset.

### 3️⃣ SMOTETomek

* Combines oversampling and cleaning techniques.
* Produces a more balanced and cleaner dataset.

---

## 🤖 Machine Learning Models Used

### Logistic Regression

* Baseline classification model.
* Fast and interpretable.

### Random Forest Classifier

* Ensemble learning algorithm.
* Handles complex fraud patterns effectively.

### Gradient Boosting Classifier

* Sequential boosting approach.
* Provides strong predictive performance.

---

## 📈 Evaluation Metrics

The models were evaluated using:

* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Average Precision Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

These metrics are especially important for highly imbalanced fraud detection problems.

---

## 📊 Visualizations Included

* Fraud vs Genuine Distribution
* Transaction Distribution Pie Chart
* Transaction Amount Analysis
* Correlation Heatmap
* Resampling Comparison
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance Analysis
* Threshold Optimization Graph

---

## 🏆 Project Highlights

✅ Real-world Credit Card Fraud Dataset

✅ Advanced Data Preprocessing

✅ Class Imbalance Handling

✅ Multiple Machine Learning Models

✅ Model Comparison and Selection

✅ Threshold Optimization

✅ Fraud Prediction Example

✅ Model Export and Reusability

✅ Professional Data Visualization

---


## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE, SMOTETomek)
* Joblib
* KaggleHub

---

## 💡 Business Impact

An effective fraud detection system helps financial institutions:

* Reduce financial losses.
* Detect suspicious activities in real-time.
* Improve customer trust and security.
* Minimize false fraud alerts.
* Strengthen risk management systems.

---

## 👨‍💻 Author

**Salik Ahamad Shabbir Khan**

Data Science Intern @ CodSoft




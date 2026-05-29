# 🎬 Task 2: Movie Rating Prediction with Python

## 📖 Project Overview

This project predicts movie ratings using machine learning techniques based on movie attributes such as genre, director, cast members, duration, release year, and audience votes.

The objective is to analyze historical IMDb movie data and build a regression model capable of estimating movie ratings with high accuracy.

This project was completed as part of the **CodSoft Data Science Internship Program**.

---

## 🚀 Core Engineering Features

### 🧹 Data Cleaning
- Removed movies without ratings
- Handled missing values
- Converted duration into numerical minutes
- Cleaned year and vote information

### ⚙️ Advanced Feature Engineering
- Genre Count
- Primary Genre Extraction
- Director Frequency Encoding
- Actor Popularity Score
- Cast Popularity Index
- Decade-Based Features
- Movie Age Features
- Log-Transformed Votes

### 📊 Exploratory Data Analysis
- Rating Distribution
- Average Rating by Decade
- Genre Analysis
- Votes vs Rating Relationship
- Duration vs Rating Relationship
- Genre Count Analysis

---

## 🤖 Machine Learning Models

The following regression models were evaluated:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor

Model performance was validated using **5-Fold Cross Validation**.

---

## 🔍 Hyperparameter Optimization

Gradient Boosting Regressor was optimized using:

- GridSearchCV
- Cross Validation
- RMSE Scoring

---

## 📈 Evaluation Metrics

The project uses:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

These metrics provide comprehensive evaluation of regression model performance.

---

## 📉 Visualizations

The notebook includes:

- Rating Distribution Histogram
- Decade Rating Trends
- Genre Performance Analysis
- Votes vs Rating Scatter Plot
- Duration vs Rating Analysis
- Actual vs Predicted Ratings
- Residual Distribution
- Feature Importance Analysis

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🎯 Conclusion

A complete machine learning pipeline was developed to predict movie ratings using IMDb movie data.

The project demonstrates:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Regression Modeling
- Hyperparameter Tuning
- Model Evaluation

This project highlights practical applications of machine learning in the entertainment industry and showcases end-to-end Data Science workflow implementation.

---

### 👨‍💻 Author

**Salik Ahamad Shabbir Khan**

Data Science Intern — CodSoft

# Task 1: Titanic Survival Prediction — Enhanced Pipeline

### 📊 Project Overview
This repository contains a production-grade machine learning pipeline engineered to predict passenger survival outcomes from the classic Titanic dataset as part of the CodSoft Data Science Internship.

### ⚙️ Core Engineering Features
* **Intelligent Imputation:** Rather than using basic global column medians, missing passenger ages are dynamically calculated based on social title groupings (`Mr.`, `Miss.`, `Mrs.`, `Master.`).
* **Advanced Feature Engineering:** Extracted passenger cabin structures to map deck layers (`Deck_A` through `Deck_E`) and formulated custom behavioral categories (`IsAlone`, `FamilyCategory`, `FareBand`).
* **Robust Modeling:** Implemented an optimized **Gradient Boosting Classifier** tuned through a 5-Fold `StratifiedKFold` Grid Search cross-validation strategy.

### 📈 Final Performance Diagnostics
* **Test Accuracy:** 80.45%
* **ROC-AUC Score:** 0.8411
* **Cross-Validation ROC-AUC Mean:** 0.8798 ± 0.0303

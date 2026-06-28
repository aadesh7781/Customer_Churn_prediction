# 📊 Customer Churn Prediction using Machine Learning

<p align="center">

**An End-to-End Machine Learning Project for Predicting Telecom Customer Churn**

Data Cleaning • Exploratory Data Analysis • Feature Engineering • Model Comparison • Hyperparameter Tuning • Model Deployment

</p>

---

## 📌 Project Overview

Customer churn is one of the most critical business challenges faced by telecom companies. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project focuses on predicting whether a customer is likely to churn using supervised machine learning techniques.

The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), statistical feature selection, model comparison, hyperparameter tuning, and final model selection.

---

## 📸 Project Banner

> **Add Screenshot Here**

**Recommended Image:**

* A clean banner showing the project title with telecom/customer churn graphics.
* Name the image: `banner.png`

```text
screenshots/banner.png
```

---

# 📂 Dataset

* **Dataset:** Telco Customer Churn Dataset
* **Records:** 7,032 Customers
* **Features:** Customer demographics, account information, subscribed services, billing details, and churn status.

Target Variable:

```
Churn Value

0 → Customer Stayed
1 → Customer Churned
```

---

# 🎯 Project Objectives

* Predict customer churn using machine learning.
* Identify the most influential factors affecting churn.
* Compare multiple machine learning algorithms.
* Select the best-performing model through systematic evaluation.
* Prepare the model for deployment.

---

# 🛠️ Technologies Used

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Data Manipulation    | Pandas, NumPy       |
| Data Visualization   | Matplotlib, Seaborn |
| Machine Learning     | Scikit-Learn        |
| Model Serialization  | Joblib              |
| Notebook             | Jupyter Notebook    |

---

# 📈 Machine Learning Workflow

```text
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Selection
        │
        ▼
Encoding
        │
        ▼
Train-Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Final Model
```

---

# 📸 Workflow Diagram

> **Add Screenshot Here**

Recommended image:

```
workflow.png
```

A screenshot of your notebook workflow or a custom ML pipeline diagram.

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Converted `Total Charges` into numeric values.
* Removed invalid records containing missing values.
* Removed identifier columns.
* Removed leakage features.
* Investigated high-cardinality features.
* Performed feature engineering.
* Encoded categorical variables.
* Standardized numerical variables where required.

---

# 📊 Exploratory Data Analysis

EDA was performed to understand customer behavior and identify meaningful predictors.

Analysis included:

* Class imbalance analysis
* Correlation analysis
* Heatmap visualization
* Geographic feature analysis
* Chi-Square statistical testing
* Feature importance investigation

---

# 📸 EDA Visualizations

### Target Variable Distribution

> Add Screenshot

```
target_distribution.png
```

---

### Correlation Heatmap

> Add Screenshot

```
heatmap.png
```

---

### ROC Curve

> Add Screenshot

```
roc_curve.png
```

---

### Precision Recall Curve

> Add Screenshot

```
precision_recall_curve.png
```

---

### Confusion Matrix

> Add Screenshot

```
confusion_matrix.png
```

---

### Feature Importance

> Add Screenshot

```
feature_importance.png
```

---

# 🔍 Feature Selection

Feature selection was performed using:

* Exploratory Data Analysis
* Correlation Analysis
* Chi-Square Test
* Business Knowledge
* Model Performance Comparison

Three feature sets were evaluated:

| Dataset   | Description         |
| --------- | ------------------- |
| Dataset 1 | All Features        |
| Dataset 2 | Without City        |
| Dataset 3 | Without City & CLTV |

This approach ensured that feature removal decisions were based on experimental evidence rather than assumptions.

---

# 🤖 Machine Learning Models Evaluated

The following supervised learning algorithms were compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting

---

# 📈 Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

Cross Validation was performed to evaluate model stability.

---

# ⚙️ Hyperparameter Tuning

The best-performing models were optimized using **GridSearchCV**.

### Logistic Regression

Optimized Parameters

```python
C = 0.001
Penalty = "l2"
Solver = "liblinear"
```

### Gradient Boosting

Optimized Parameters

```python
Learning Rate = 0.05
Max Depth = 4
Estimators = 200
Subsample = 0.8
```

---

# 🏆 Final Model

After evaluating all models, **Logistic Regression** was selected as the final model.

Reasons:

* Highest F1 Score
* Strong Recall
* Competitive ROC-AUC
* Fast training
* Easy interpretability
* Stable Cross Validation performance

---

# 📊 Final Results

| Model                       | Accuracy   | Precision  | Recall     | F1 Score   | ROC-AUC    |
| --------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression (Tuned) | **79.74%** | **60.32%** | **69.52%** | **64.60%** | **0.8384** |
| Gradient Boosting (Tuned)   | 79.25%     | 62.50%     | 54.81%     | 58.40%     | 0.8456     |

---

# 📂 Project Structure

```
Customer-Churn-Prediction/

│
├── data/
│
├── notebook/
│     customer_churn_prediction.ipynb
│
├── models/
│     telco_churn_model.pkl
│     scaler.pkl
│
├── screenshots/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── report.pdf
```

---

# 🚀 Future Improvements

* Deploy using Streamlit
* Integrate with Flask/FastAPI
* Experiment with XGBoost
* Handle imbalance using SMOTE
* Explain predictions using SHAP
* Deploy on Streamlit Community Cloud

---

# 📬 Contact

**Aadesh Sharma**

LinkedIn: *Add your LinkedIn profile*

GitHub: *Add your GitHub profile*

Email: Aadeshsharma1905@gmail.com

---

# ⭐ If you found this project useful, consider giving it a Star!

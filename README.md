# White-Box-Supervised-ML Classification & Regression  

## 📌 Overview  
This repository showcases two complete end-to-end machine learning projects:  

1. **Credit Scoring Classification Model** — predicting a customer’s credit score category (Good, Standard, or Poor) based on financial behavior and demographic features.  
2. **Bank Marketing Regression Model** — predicting a continuous outcome variable (e.g., customer deposit likelihood or campaign success rate) based on marketing and customer data.  

Each project includes full stages of data processing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and interpretation.

---

---

## 🔍 Project 1: Credit Scoring Classification  

### 🎯 Problem Statement  
A large financial institution aims to improve its **credit risk assessment** by automating the process of assigning credit scores to customers.  
The goal is to build an **explainable (white-box)** classification model to replace the manual credit rating process, reduce bias, and enhance efficiency.

### 🧩 Objectives  
- Identify key factors influencing a customer’s credit score.  
- Test multiple classification models (Logistic Regression, KNN).  
- Evaluate and compare models using accuracy, precision, recall, and F1 score.  
- Visualize confusion matrices for better interpretability.

### 📊 Dataset Highlights  
- Features include income, loans, payment delays, number of credit cards, and outstanding debt.  
- Target variable: **Credit_Score** (`Good`, `Standard`, `Poor`).  

### ⚙️ Modeling & Evaluation  
- **Models Used:** Logistic Regression, K-Nearest Neighbors (KNN).  
- **Metrics:** Accuracy, Precision, Recall, F1 Score.  
- **Visuals:**  
  - Correlation Heatmap  
  - Feature Distribution Plots  
  - Confusion Matrices  

### 🏆 Final Insights  
- **Best Model:** KNN (Selected Features) — Accuracy: ~67%.  
- **Key Predictors:** Monthly Inhand Salary, Number of Loans, Delay from Due Date, and Outstanding Debt.  
- **Business Takeaway:** Automating credit scoring improves consistency and transparency.  

---

## 📉 Project 2: Bank Marketing Regression  

### 🎯 Problem Statement  
A banking organization seeks to predict the success rate of marketing campaigns to identify customers most likely to subscribe to a term deposit.  

### 🧩 Objectives  
- Build regression models to predict campaign outcomes.  
- Understand which features (age, job type, contact method, previous campaigns) drive success.  
- Compare multiple regression algorithms for interpretability and accuracy.  

### ⚙️ Modeling & Evaluation  
- **Models Used:** Linear Regression, KNN Regressor.  
- **Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score.  
- **Visuals:**  
  - Pair Plots for Feature Relationships  
  - Correlation Heatmaps  
  - Residual and Predicted vs Actual Plots  

### 🏆 Final Insights  
- **Best Model:** Linear Regression with feature selection.  
- **Key Influences:** Contact type, previous campaign outcomes, and duration of last contact.  
- **Business Impact:** Data-driven targeting increases conversion efficiency.  

---

## 📈 Key Visualizations  

| Visualization | Purpose |
|----------------|----------|
| Credit Score Distribution | Identify class balance |
| Correlation Heatmap | Examine relationships between features |
| Boxplots | Compare numerical features across classes |
| Confusion Matrix | Visualize model misclassifications |
| Residual Plot (Regression) | Evaluate model fit |

---

## 🛠️ Tech Stack  

- **Languages:** Python  
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `seaborn`,  
  `scikit-learn`, `statsmodels`, `plotly`  

---

## ⚙️ How to Run  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/your-username/Machine-Learning-Projects.git
   cd Machine-Learning-Projects


Implement advanced models (Random Forest, XGBoost).

Apply feature selection using SHAP or LIME for interpretability.

Automate EDA and model selection through pipelines.

Add dashboard visualization using Streamlit.

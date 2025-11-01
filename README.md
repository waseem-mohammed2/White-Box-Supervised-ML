# White-Box-Supervised-ML Classification & Regression  

> **Important:** The classification dataset (`train.csv`) is large and is uploaded as `train.csv.gz`.  
> **Please extract** `train.csv` from `train.csv.gz` into `White Box - Supervised ML Project Classification- DSBPT1.ipynb` before running the classification notebook.

---

## 🔎 Project 1 — Regression: Video Game Sales

### Problem statement
A gaming company wants to predict which video games will be successful globally (target: `Global_Sales`, in millions). The goal is to help prioritize development and marketing budgets. Output should be explainable to non-technical stakeholders.

### Dataset
- **File:** `vgsales.csv`  
- **Records:** ~16,598 rows (games with sales > 100k)  
- **Key columns:** `Name`, `Platform`, `Year`, `Genre`, `Publisher`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`

### Modeling approach
- **Models tested:**  
  - Linear Regression (multiple versions with different feature sets)  
  - K-Nearest Neighbors Regressor (multiple versions)
- **Evaluation metrics (regression):**
  - **R² (coefficient of determination)** — how much variance the model explains  
  - **RMSE (Root Mean Squared Error)** — penalizes larger errors  
  - **Residual analysis** & **Predicted vs Actual** plots
- **Notes on modeling decisions:**  
  - Avoid leakage: do **not** include `Global_Sales` as a predictor. You may include selected regional sales (e.g., `NA_Sales`) carefully for one model version to compare effect.  
  - Use One-Hot Encoding for nominal categorical features (Platform, Genre, Publisher when needed) to avoid implying ordinal relations.  
  - Scale numeric features when using distance-based models (KNN).

### Visualizations in the notebook
- Distribution of `Global_Sales`  
- Sales by `Platform`, `Genre`, `Publisher` (bar charts)  
- Correlation heatmap among numeric features  
- Predicted vs Actual and Residual plots for each model

---

## 🔎 Project 2 — Classification: Credit Score Classification

### Problem statement
A global bank wants an explainable model to score customers into credit brackets (`Credit_Score` = `Good` / `Standard` / `Poor`) to reduce bias and standardize credit decisions.

### Dataset
- **File (compressed):** `classification_credit_scoring/data/train.csv.gz` → contains `train.csv` (extract to same folder)  
- **Columns (sample):** `ID`, `Customer_ID`, `Month`, `Name`, `Age`, `Occupation`, `Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`, `Delay_from_due_date`, `Credit_Mix`, `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Credit_History_Age`, `Payment_of_Min_Amount`, `Total_EMI_per_month`, `Payment_Behaviour`, `Monthly_Balance`, `Credit_Score`, etc.

### Modeling approach
- **Models tested:**  
  - Logistic Regression (two versions: full features and selected features)  
  - K-Nearest Neighbors Classifier (two versions)
- **Evaluation metrics (classification):**
  - **Accuracy** — overall correct proportion  
  - **Precision** — correctness of positive predictions (per class)  
  - **Recall (Sensitivity)** — ability to find actual positives (per class)  
  - **F1 Score** — harmonic mean of precision & recall  
  - **Confusion Matrix** — visualize true vs predicted class counts  
- **Preprocessing highlights:**  
  - Remove / exclude identifying columns (`ID`, `SSN`, `Customer_ID`) from predictors.  
  - Treat or drop problematic rows (malformed ages like `-500`, non-numeric entries in numeric columns), or convert to numeric after cleaning.  
  - For nominal categorical features (e.g., `Occupation`, `Type_of_Loan`, `Payment_Behaviour`), use **One-Hot Encoding**.  
  - For ordinal-like columns (if any clear order exists), a **LabelEncoder** can be considered — but do **not** use LabelEncoder on nominal features.  
  - Scale numeric features when using KNN.  
  - Keep the pipeline simple and explainable (no black-box models).

### Visualizations in the notebook
- Class distribution (`Credit_Score`) — bar chart  
- Boxplots: `Annual_Income`, `Monthly_Inhand_Salary` by credit class  
- Correlation heatmap for numeric predictors  
- Confusion matrix heatmaps for each model (visual clarity for stakeholders)

---

## ✅ How to run the notebooks

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Machine-Learning-Projects.git
   cd Machine-Learning-Projects

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt

(requirements.txt should include: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, plotly — adjust per your environment.)

3. **Prepare the data**
    - Regression: Ensure vgsales.csv is present within the same folder.
    - Classification:
    - 'Extract' the train.csv.gz in the same folder of the notebook which will create the train.csv file.

4. **Open notebooks**
    ```bash
    jupyter notebook


- Run White Box - Supervised ML Project Classification- DSBPT1.ipynb for classification analysis.
- Run White Box - Supervised ML Project Regression- DSBPT1.ipynb for Regression analysis.

### 📌 Notes on Reproducibility & Explainability

**Every notebook includes:**

- 🧭 EDA with visualizations and short interpretations — helps understand distributions, outliers, and relationships.
- 🧹 Data cleaning steps with reasoning — explains why and how certain rows or values were corrected or removed.
- 🔁 Two model versions per algorithm — e.g., Logistic Regression v1 (basic features) and v2 (engineered or selected features) to show performance improvements.
- 📊 Evaluation section — metrics and confusion matrices for classification; R², RMSE, and residual diagnostics for regression.
- 🧾 Plain-English explanations — designed for presentation to non-technical stakeholders, keeping transparency at the core.

### 📈 Recommended Next Steps / Improvements

🔄 Try resampling or class-balancing (SMOTE / undersampling) only if necessary — and document the effect clearly.

⚙️ Add hyperparameter tuning using GridSearchCV to optimize KNN neighbors, Logistic Regression regularization, etc.

📊 Build a Streamlit or Plotly Dash dashboard for interactive visualization and stakeholder reporting.

### 🛠 Tech Stack
- Language: Python 3.x

**Main Libraries:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- plotly

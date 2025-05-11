# 🏥 Insurance Charges Predictor

A user-friendly Streamlit web application that predicts individual health insurance charges based on demographic and lifestyle information. Built using **Random Forest**, with support for other models like **Linear Regression**, **LightGBM**, and **XGBoost** for comparison.

[Live app](https://health-insurance-app-app-zks4tkjnbak6wcycvr5b76.streamlit.app/) <!-- Replace with your actual image URL -->

---

## 📌 Features

- 📋 User input for age, gender, BMI, smoking status, number of children, and region.
- ⚙️ Feature engineering for enhanced prediction accuracy (`age_group`, `bmi_smoker`, `age_bmi`, etc.).
- 📈 Trained and evaluated models:
  - Linear Regression
  - Random Forest Regressor (Base and Tuned)
  - LightGBM
  - XGBoost
- 🎯 Performance metrics (RMSE, R², Adjusted R²)
- 📊 Visualizations for distribution, boxplots, correlations, and feature importance.
- 🧠 Machine learning models trained using `scikit-learn`, `xgboost`, and `lightgbm`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/obed254github/health-insurance-streamlit-app.git
cd health-insurance-streamlit-app
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

> Make sure `joblib`, `streamlit`, `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, and `numpy` are installed.

### 3. Run the app

```bash
streamlit run data_analysis.py
```

---

## 🧠 How It Works

### Input Features:

- `age`: Age of the individual
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of dependents
- `smoker`: Smoker status (yes/no)
- `region`: Geographic region in the U.S.

### Engineered Features:

- `age_group`: Categorical (Young, Middle-aged, Senior)
- `bmi_smoker`: Interaction term between BMI and smoking status
- `age_smoker`: Age × Smoker
- `age_bmi`: Age × BMI
- `children_per_age`: Children divided by age

### Output:

- Predicted Insurance Charges in USD.

---

## 📦 Directory Structure

```
.
├── app.py
├── utils/
│   ├── final_model.pkl
│   └── preprocessor.pkl
├── data/
│   └── HealthData.csv
├── pages/
    └── Predictive_model.py
├── requirements.txt
└── README.md
```

---

## 📊 Model Evaluation Summary

| Model                   | R² Score  | Adjusted R² | RMSE        |
| ----------------------- | --------- | ----------- | ----------- |
| Linear Regression       | 0.865     | 0.859       | 4577.96     |
| Base Random Forest      | 0.870     | 0.863       | 4509.22     |
| **Tuned Random Forest** | **0.880** | **0.875**   | **4325.11** |
| LightGBM                | 0.880     | 0.870       | 4400.60     |
| XGBoost                 | 0.870     | 0.867       | 4442.12     |

---

## 💡 Future Improvements

- Add CSV/JSON export for predictions
- Model interpretability using SHAP or LIME
- Responsive design for mobile devices

---

## 👨‍💻 Author

**Obadiah Kiptoo**  
_MSc in Data Science & Analytics Grand Valley State University_  
🔗 [LinkedIn](https://www.linkedin.com/in/obadiah-kiptoo-85480b175/)  
📧 obadiahkiptoo1998@gmail.com

---

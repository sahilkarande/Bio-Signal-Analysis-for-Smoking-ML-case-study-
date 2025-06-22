# Smoking Classification Using Bio-Signal Data

##  Project Objective
This project aims to develop a machine learning model that classifies individuals as smokers or non-smokers using health checkup and biological signal data. The model is intended to assist in proactive health monitoring and analytics within healthcare environments.

---

##  Dataset Overview
The dataset consists of approximately **55,000 records** and **27 attributes**, including both numerical and categorical health features:

- **Demographics**: Gender, Age
- **Body Measurements**: Height, Weight, Waist circumference
- **Sensory Tests**: Eyesight (left/right), Hearing (left/right)
- **Vitals & Tests**: Blood Pressure, Blood Sugar, Cholesterol (HDL, LDL), Triglyceride
- **Liver Enzymes**: AST, ALT, GTP
- **Kidney/Blood Indicators**: Hemoglobin, Urine Protein, Serum Creatinine
- **Oral Health**: Dental Caries, Tartar
- **Target**: `smoking` (0 = Non-smoker, 1 = Smoker)

---

##  Data Preprocessing

- **Null Handling**: Checked and cleaned missing values.
- **Encoding**: Converted categorical columns (e.g., gender, oral health) into numerical formats.
- **Outlier Removal**: Applied **IQR filtering** on 22 numerical columns to eliminate extreme outliers and improve model stability.
- **Feature Scaling**: StandardScaler was applied to continuous variables for uniformity.

---

##  Exploratory Data Analysis (EDA)

### Distribution Insights
- Most individuals are aged between **30–60 years**, with a peak around **40–50**.
- **Male** entries are dominant in the dataset.
- Height and weight show typical adult distributions (e.g., 160–180 cm, 60–80 kg).
- Cholesterol, blood pressure, and triglyceride levels mostly lie within normal health ranges.
- Tartar and dental caries show visible differences between smokers and non-smokers.

### Smoking Distribution
- **Non-Smokers**: 63% (≈35,200 records)
- **Smokers**: 37% (≈20,400 records)

---

##  Correlation Analysis

Key correlations with the `smoking` variable:

| Feature            | Correlation |
|--------------------|-------------|
| Gender             | 0.51        |
| Hemoglobin         | 0.40        |
| Height             | 0.40        |
| Weight             | 0.30        |
| Triglyceride       | 0.25        |
| GTP (Liver Enzyme) | 0.24        |
| HDL                | -0.18       |
| Age                | -0.16       |

These correlations suggest that smoking is more common in males with elevated hemoglobin, weight, and liver enzyme levels.

---

##  Feature Selection

- Used `SelectKBest` with **ANOVA F-test** to rank features.
- Top 10 selected features:
  - Hemoglobin, Height, Weight, Waist, Gender, HDL, GTP, Serum Creatinine, Triglyceride, LDL

---

##  Machine Learning Models

Three classification models were trained and evaluated:

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.79    | ~0.84   |
| Decision Tree       | ~0.76    | ~0.81   |
| Random Forest       | ~0.84    | ~0.88   |

- **Random Forest** emerged as the most effective, achieving the highest accuracy and ROC-AUC.
- Evaluation also included precision, recall, F1-score, and confusion matrices.

---

##  Conclusion

- The **Random Forest classifier** is recommended for predicting smoking habits based on routine health indicators.
- Strong predictors include **gender**, **hemoglobin**, **height**, **GTP**, and **triglyceride** levels.
- The model is interpretable and deployable in real-world health screening applications.

---

## Tech Stack

- Python 3.x
- pandas, numpy
- seaborn, matplotlib
- scikit-learn

---

## Project Structure

```

Smoking\_Classification/
├── casestudy.py          # Main script with preprocessing, modeling, and evaluation
├── README.md             # Project documentation
├── smoking\_data.csv      # Input dataset (not uploaded for privacy)
├── models/               # Trained model pickle files (optional)
└── notebooks/            # Jupyter analysis notebooks (optional)

```

---

## Future Improvements

- Address gender imbalance for fairer model generalization.
- Integrate SHAP or LIME for feature importance interpretation.
- Try ensemble stacking or advanced boosting models like XGBoost or CatBoost.
- Develop a Streamlit-based frontend to interact with predictions.


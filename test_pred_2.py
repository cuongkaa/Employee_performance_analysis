import pandas as pd
import numpy as np
import warnings
from src.data.loader import load_data, load_config
from src.data.cleaner import HRDataCleaner
from src.features.builder import FeatureBuilder
from src.models.supervised import train_random_forest

warnings.filterwarnings('ignore')

config = load_config("configs/params.yaml")
df = pd.read_csv("data/raw/HR_Analytics.csv")
cleaner = HRDataCleaner(target_col="Attrition")
df_clean = cleaner.clean(df)

df_enc = cleaner.encode(FeatureBuilder(df_clean.copy()).build_all())
X = df_enc.drop(columns=["Attrition"])
y = df_enc["Attrition"]

model = train_random_forest(X, y, n_estimators=100, random_state=42)

def predict_attrition(age, income, distance, js, es, wlb, overtime, y_company, t_years, stock, marital):
    sample_data = df_clean.drop(columns=["Attrition"], errors="ignore").mode().iloc[0:1].copy()
    sample_data["Age"] = age
    sample_data["MonthlyIncome"] = income
    sample_data["DistanceFromHome"] = distance
    sample_data["JobSatisfaction"] = js
    sample_data["EnvironmentSatisfaction"] = es
    sample_data["WorkLifeBalance"] = wlb
    sample_data["YearsAtCompany"] = y_company
    sample_data["TotalWorkingYears"] = t_years
    sample_data["OverTime"] = overtime
    sample_data["StockOptionLevel"] = stock
    sample_data["MaritalStatus"] = marital

    fb_custom = FeatureBuilder(sample_data)
    sample_featured = fb_custom.build_all()
    sample_enc = cleaner.encode(sample_featured)

    input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
    for col in sample_enc.columns:
        if col in input_data.columns:
            input_data[col] = sample_enc[col].values[0]

    prob = model.predict_proba(input_data)[0][1]
    return prob

print("Baseline mode():", predict_attrition(30, 5000, 5, 3, 3, 3, "No", 5, 10, 1, "Married"))
print("User 1 (Married):", predict_attrition(25, 100, 30, 1, 1, 1, "Yes", 1, 10, 0, "Married"))
print("User 2 (Single):", predict_attrition(25, 100, 30, 1, 1, 1, "Yes", 10, 10, 0, "Single"))
print("User 3 (Single, 1 yr at company):", predict_attrition(25, 100, 30, 1, 1, 1, "Yes", 1, 1, 0, "Single"))
print("User 4 (Young, new, single):", predict_attrition(18, 100, 30, 1, 1, 1, "Yes", 0, 0, 0, "Single"))

# Print top 5 features according to RF importances
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features:")
print(importances.head(10))

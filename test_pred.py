import pandas as pd
import numpy as np
from src.data.loader import load_data, load_config
from src.data.cleaner import HRDataCleaner
from src.features.builder import FeatureBuilder
from src.models.supervised import train_random_forest

config = load_config("configs/params.yaml")
df = pd.read_csv("data/raw/HR_Analytics.csv")
cleaner = HRDataCleaner(target_col="Attrition")
df_clean = cleaner.clean(df)

df_enc = cleaner.encode(FeatureBuilder(df_clean.copy()).build_all())
X = df_enc.drop(columns=["Attrition"])
y = df_enc["Attrition"]

model = train_random_forest(X, y, n_estimators=100, random_state=42)

sample_data = df_clean.drop(columns=["Attrition"], errors="ignore").mode().iloc[0:1].copy()
sample_data["Age"] = 18
sample_data["MonthlyIncome"] = 1000
sample_data["DistanceFromHome"] = 30
sample_data["JobSatisfaction"] = 1
sample_data["EnvironmentSatisfaction"] = 1
sample_data["WorkLifeBalance"] = 1
sample_data["YearsAtCompany"] = 1
sample_data["TotalWorkingYears"] = 1
sample_data["OverTime"] = "Yes"

fb_custom = FeatureBuilder(sample_data)
sample_featured = fb_custom.build_all()
sample_enc = cleaner.encode(sample_featured)

input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
for col in sample_enc.columns:
    if col in input_data.columns:
        input_data[col] = sample_enc[col].values[0]

prob = model.predict_proba(input_data)[0][1]
print("Probability:", prob)

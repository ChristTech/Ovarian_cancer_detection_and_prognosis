import pandas as pd
import joblib

# Load the model
model = joblib.load("multi_risk_model.sav")

# Load the same dataset or new samples
df = pd.read_csv("csv/SurvivalNormalBoth.csv")

# Preprocess input
df = df.drop(columns=["Unnamed: 0", "Cancer.Type"])
for col in ["SURVIVAL", "PROGRESSION", "RECURRENCE"]:
    df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# Prepare features (X) only
X = df.drop(columns=["SURVIVAL", "PROGRESSION", "RECURRENCE"])
X = X.fillna(X.mean())

# Predict
predictions = model.predict(X)

# View predictions
pred_df = pd.DataFrame(predictions, columns=["SURVIVAL_PRED", "PROGRESSION_PRED", "RECURRENCE_PRED"])
print(pred_df.head())


results = pd.concat([df[["SURVIVAL", "PROGRESSION", "RECURRENCE"]].reset_index(drop=True), pred_df], axis=1)
print(results.head())

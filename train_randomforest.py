import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("csv/SurvivalNormalBoth.csv")

# Drop unneeded columns
df = df.drop(columns=["Unnamed: 0", "Cancer.Type"], errors='ignore')

# Encode Figo.Stage
stage_mapping = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1, "Stage IC": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
    "Stage IV": 4
}
df["Figo.Stage"] = df["Figo.Stage"].map(stage_mapping)

target_cols = ["SURVIVAL", "PROGRESSION", "RECURRENCE"]

# Convert to strings, then map to 1/0
for col in target_cols:
    df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "UNKNOWN": 0, "NA": 0})

# Drop rows with missing target values
df = df.dropna(subset=target_cols)

# Split into features and targets
X = df.drop(columns=target_cols + ["Patient", "Status (Alive)", "Days Survived (7300 Max)", "ICD 10 Code", "Age Days", "Race", "Neoplasm Histologic Grade", "Tumor Residual Disease", "Lymphatic Invasion", "Total Dose", "Total Dose Units", "Number Cycles", "Start Date", "End Date", "Therapy Type", "Drug", "Regimen Indication"], errors='ignore')
y = df[target_cols]

# Fill missing values in features
X = X.fillna(X.mean())

# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler.feature_names_in_ = X.columns.tolist()  # Ensure scaler stores feature names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, y_train)  # Fit with unscaled X to preserve feature names

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# Save model and scaler
joblib.dump(model, "multi_risk_model.sav")
joblib.dump(scaler, "scaler.sav")
print("✅ Model saved as 'multi_risk_model.sav'")
print("✅ Scaler saved as 'scaler.sav'")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("csv/SurvivalNormalBoth.csv")

# Drop unneeded columns
df = df.drop(columns=["Unnamed: 0", "Cancer.Type"])

target_cols = ["SURVIVAL", "PROGRESSION", "RECURRENCE"]

# Convert to strings, then map to 1/0
for col in target_cols:
    df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# Drop rows with missing target values
df = df.dropna(subset=target_cols)

# Split into features and targets
X = df.drop(columns=target_cols)
y = df[target_cols]

# Fill missing values in features
X = X.fillna(X.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X.fillna(X.mean()))
joblib.dump(scaler, "scaler.sav")


# Save model
joblib.dump(model, "multi_risk_model.sav")
print("✅ Model saved as 'multi_risk_model.sav'")

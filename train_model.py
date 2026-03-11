import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("diabetes.csv")

# Remove SkinThickness column if exists
if "SkinThickness" in df.columns:
    df = df.drop(columns=["SkinThickness"])

# Define features
X = df[[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]]

y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model retrained successfully with 7 features!")
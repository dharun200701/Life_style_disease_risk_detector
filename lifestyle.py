import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Features to use
FEATURES = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Blood Pressure",
    "Daily Steps",
]

TARGET = "Sleep Disorder"


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

df = df[FEATURES + [TARGET]].copy()


# -----------------------------
# Encode categorical columns
# -----------------------------
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le


X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("\nModel trained successfully!")
print("\nEnter patient details:\n")


# -----------------------------
# USER INPUT
# -----------------------------
user_data = {}

for col in FEATURES:
    value = input(f"{col}: ")

    if col in encoders:
        le = encoders[col]
        if value not in le.classes_:
            print("Unknown value. Using default.")
            value = le.classes_[0]
        value = le.transform([value])[0]
    else:
        value = float(value)

    user_data[col] = value


user_df = pd.DataFrame([user_data])

prediction = model.predict(user_df)[0]

# Decode prediction
prediction = encoders[TARGET].inverse_transform([prediction])[0]

print("\nPredicted  Disorder:", prediction)

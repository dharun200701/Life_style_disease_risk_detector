import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# ==========================================================
# LOAD DATASET
# ==========================================================

sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
synthetic_df = pd.read_csv("synthetic_health_lifestyle_dataset.csv")

df = pd.concat(
    [sleep_df, synthetic_df],
    ignore_index=True
)


# ==========================================================
# REMOVE MISSING TARGET
# ==========================================================

df.dropna(
    subset=["Sleep Disorder"],
    inplace=True
)


# ==========================================================
# SPLIT BLOOD PRESSURE
# ==========================================================

if "Blood Pressure" in df.columns:

    df[["Systolic", "Diastolic"]] = (
        df["Blood Pressure"]
        .str.split("/", expand=True)
    )

    df["Systolic"] = pd.to_numeric(
        df["Systolic"],
        errors="coerce"
    )

    df["Diastolic"] = pd.to_numeric(
        df["Diastolic"],
        errors="coerce"
    )

    df.drop(
        columns=["Blood Pressure"],
        inplace=True
    )


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "Age",
    "Gender",
    "Sleep Duration",
    "Physical Activity Level",
    "BMI Category",
    "Systolic",
    "Diastolic"
]

target = "Sleep Disorder"


# ==========================================================
# LABEL ENCODING
# ==========================================================

encoders = {}

for col in df.select_dtypes(
    include=["object"]
).columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

for col in features:

    if pd.api.types.is_numeric_dtype(df[col]):

        df[col] = df[col].fillna(
            df[col].median()
        )

    else:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )


# ==========================================================
# TRAINING DATA
# ==========================================================

X = df[features]
y = df[target]


# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "sleep_disorder_model.pkl"
)


# ==========================================================
# SAVE ENCODERS + FEATURES
# ==========================================================

metadata = {
    "encoders": encoders,
    "features": features,
    "target": target
}

joblib.dump(
    metadata,
    "model_metadata.pkl"
)


print("======================================")
print("MODEL TRAINING COMPLETED")
print("======================================")
print("Model saved as:")
print("sleep_disorder_model.pkl")
print()
print("Metadata saved as:")
print("model_metadata.pkl")
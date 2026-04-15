import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -------------------------------
# Load and Prepare Dataset
# -------------------------------

sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
synthetic_df = pd.read_csv("synthetic_health_lifestyle_dataset.csv")

df = pd.concat([sleep_df, synthetic_df], ignore_index=True)

# Remove rows without target
df.dropna(subset=["Sleep Disorder"], inplace=True)

# Split Blood Pressure
if "Blood Pressure" in df.columns:
    df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
    df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)

# Features
features = [
    "Age","Gender","Sleep Duration","Physical Activity Level",
    "BMI Category","Systolic","Diastolic"
]

target = "Sleep Disorder"

# Encode categorical data
encoders = {}
for col in df.select_dtypes(include=["object"]):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Fill missing values
for col in features:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Train Model
X = df[features]
y = df[target]

model = RandomForestClassifier()
model.fit(X, y)

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    data = request.form.to_dict()
    user_data = pd.DataFrame([data])

    # Convert numeric fields
    numeric_cols = ["Age","Sleep Duration","Physical Activity Level","Systolic","Diastolic"]
    for col in numeric_cols:
        if col in user_data:
            user_data[col] = pd.to_numeric(user_data[col], errors='coerce')

    # Encode categorical
    for col in user_data.select_dtypes(include=['object']).columns:
        if col in encoders:
            le = encoders[col]
            user_data[col] = user_data[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    # Ensure all features exist
    for col in features:
        if col not in user_data:
            user_data[col] = 0

    user_data = user_data[features]

    # Prediction
    pred_encoded = model.predict(user_data)[0]
    prediction = encoders[target].inverse_transform([pred_encoded])[0]

    # -------------------------------
    # Reasons + Tips
    # -------------------------------

    reasons = []
    tips = []

    sleep = float(data.get("Sleep Duration", 0))
    activity = float(data.get("Physical Activity Level", 0))
    systolic = float(data.get("Systolic", 120))

    if sleep < 6:
        reasons.append("Low sleep duration")
        tips.append("Sleep at least 7–8 hours daily")

    if activity < 20:
        reasons.append("Low physical activity")
        tips.append("Do at least 30 minutes of exercise")

    if systolic > 130:
        reasons.append("High blood pressure")
        tips.append("Reduce salt and manage stress")

    if not reasons:
        reasons.append("Healthy lifestyle")
        tips.append("Maintain your routine")

    # -------------------------------
    # Risk Level + Score
    # -------------------------------

    if prediction.lower() in ["none", "normal"]:
        risk = "Low"
        color = "green"
        score = 80
    elif prediction.lower() in ["insomnia"]:
        risk = "Medium"
        color = "orange"
        score = 50
    else:
        risk = "High"
        color = "red"
        score = 20

    return render_template(
        "result.html",
        prediction=prediction,
        reasons=reasons,
        tips=tips,
        risk=risk,
        color=color,
        score=score
    )


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
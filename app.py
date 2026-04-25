# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import shap
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ==============================
# INIT APP
# ==============================
app = Flask(__name__)

# ==============================
# LOAD DATA
# ==============================
sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
synthetic_df = pd.read_csv("synthetic_health_lifestyle_dataset.csv")

df = pd.concat([sleep_df, synthetic_df], ignore_index=True)

df.dropna(subset=["Sleep Disorder"], inplace=True)

# Split BP
if "Blood Pressure" in df.columns:
    df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
    df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)

# ==============================
# FEATURES
# ==============================
features = [
    "Age","Gender","Sleep Duration",
    "Physical Activity Level","BMI Category",
    "Systolic","Diastolic"
]

target = "Sleep Disorder"

# ==============================
# ENCODING
# ==============================
encoders = {}
for col in df.select_dtypes(include=["object"]):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Fill missing
for col in features:
    df[col] = df[col].fillna(df[col].median())

# ==============================
# TRAIN MODEL
# ==============================
X = df[features]
y = df[target]

model = RandomForestClassifier()
model.fit(X, y)

# ==============================
# SHAP EXPLAINER
# ==============================
explainer = shap.TreeExplainer(model)

# ==============================
# ROUTES
# ==============================

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    data = request.form.to_dict()
    user_data = pd.DataFrame([data])

    # Convert numeric
    numeric_cols = ["Age","Sleep Duration","Physical Activity Level","Systolic","Diastolic"]
    for col in numeric_cols:
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

    # ==============================
    # PREDICTION
    # ==============================
    pred_encoded = model.predict(user_data)[0]
    prediction = encoders[target].inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(user_data)[0]
    confidence = max(probs)

    # ==============================
    # SHAP EXPLANATION
    # ==============================
    shap_values = explainer.shap_values(user_data)
    if isinstance(shap_values, list):
        class_index = min(int(pred_encoded), len(shap_values) - 1)
        shap_val = shap_values[class_index][0]
    elif getattr(shap_values, "ndim", 0) == 3:
        # Newer SHAP can return (samples, features, classes)
        class_index = min(int(pred_encoded), shap_values.shape[2] - 1)
        shap_val = shap_values[0, :, class_index]
    else:
        shap_val = shap_values[0]

    feature_names = user_data.columns
    feature_impact = list(zip(feature_names, shap_val))

    # Sort by importance
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    # ==============================
    # REASONS (Top 4)
    # ==============================
    reasons = []
    for feature, val in feature_impact[:4]:
        if val > 0:
            reasons.append(f"{feature} is contributing to increased risk based on your current lifestyle pattern.")
        else:
            reasons.append(f"{feature} is helping reduce the risk due to healthy values.")

    # ==============================
    # RECOMMENDATIONS
    # ==============================
    tips = []

    if confidence > 0.7:
        tips.append("Your risk is relatively high. Immediate lifestyle improvements are recommended.")
    else:
        tips.append("Your risk is moderate. Small improvements can significantly reduce future risk.")

    tips.append("Maintain 7–8 hours of consistent sleep daily.")
    tips.append("Increase physical activity like walking, jogging, or exercise.")
    tips.append("Maintain a healthy BMI through balanced diet.")
    tips.append("Monitor blood pressure and manage stress effectively.")

    # ==============================
    # RISK LEVEL + COLOR
    # ==============================
    if prediction.lower() in ["none", "normal"]:
        risk = "Low"
        color = "green"
        score = int(confidence * 40)

    elif prediction.lower() == "insomnia":
        risk = "Medium"
        color = "orange"
        score = int(confidence * 70)

    else:
        risk = "High"
        color = "red"
        score = int(confidence * 100)

    # ==============================
    # RENDER RESULT PAGE
    # ==============================
    return render_template(
        "result.html",
        prediction=prediction,
        risk=risk,
        color=color,
        score=score,
        reasons=reasons,
        tips=tips
    )


# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
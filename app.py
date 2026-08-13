# ============================================================
# LIFESTYLE DISEASE / SLEEP DISORDER PREDICTION SYSTEM
# Random Forest + SHAP Explainable AI
# ============================================================

import pandas as pd
import numpy as np
import shap

from flask import Flask, render_template, request

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from groq_chatbot import chat_with_groq

# ============================================================
# INITIALIZE FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD DATASET
# ============================================================

sleep_df = pd.read_csv(
    "Sleep_health_and_lifestyle_dataset.csv"
)

synthetic_df = pd.read_csv(
    "synthetic_health_lifestyle_dataset.csv"
)

df = pd.concat(
    [sleep_df, synthetic_df],
    ignore_index=True
)


# ============================================================
# REMOVE MISSING TARGET
# ============================================================

df.dropna(
    subset=["Sleep Disorder"],
    inplace=True
)


# ============================================================
# SPLIT BLOOD PRESSURE
# ============================================================

if "Blood Pressure" in df.columns:

    df[["Systolic", "Diastolic"]] = (
        df["Blood Pressure"]
        .astype(str)
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


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col for col in features + [target]
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

encoders = {}

for col in df.select_dtypes(
    include=["object"]
).columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for col in features:

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):

        df[col] = df[col].fillna(
            df[col].median()
        )

    else:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )


# ============================================================
# TRAINING DATA
# ============================================================

X = df[features]

y = df[target]


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X, y)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# FEATURE DISPLAY NAMES
# ============================================================

feature_display_names = {
    "Age": "Age",
    "Gender": "Gender",
    "Sleep Duration": "Sleep Duration",
    "Physical Activity Level": "Physical Activity",
    "BMI Category": "BMI Category",
    "Systolic": "Systolic BP",
    "Diastolic": "Diastolic BP"
}


# ============================================================
# ROUTE: HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ROUTE: PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # GET USER DATA
    # --------------------------------------------------------

    data = request.form.to_dict()

    user_data = pd.DataFrame(
        [data]
    )


    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_cols = [
        "Age",
        "Sleep Duration",
        "Physical Activity Level",
        "Systolic",
        "Diastolic"
    ]

    for col in numeric_cols:

        user_data[col] = pd.to_numeric(
            user_data[col],
            errors="coerce"
        )


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_cols = [
        "Gender",
        "BMI Category"
    ]

    for col in categorical_cols:

        if col in encoders:

            le = encoders[col]

            value = str(
                user_data[col].iloc[0]
            )

            if value in le.classes_:

                user_data[col] = (
                    le.transform([value])[0]
                )

            else:

                user_data[col] = 0


    # --------------------------------------------------------
    # MAKE SURE ALL FEATURES EXIST
    # --------------------------------------------------------

    for col in features:

        if col not in user_data.columns:

            user_data[col] = 0


    # --------------------------------------------------------
    # SELECT FEATURES IN SAME ORDER
    # --------------------------------------------------------

    user_data = user_data[
        features
    ]


    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    for col in features:

        if user_data[col].isnull().any():

            user_data[col] = (
                user_data[col].fillna(
                    X[col].median()
                )
            )


    # ========================================================
    # PREDICTION
    # ========================================================

    pred_encoded = model.predict(
        user_data
    )[0]


    # --------------------------------------------------------
    # DECODE PREDICTION
    # --------------------------------------------------------

    target_encoder = encoders[target]

    prediction = (
        target_encoder
        .inverse_transform(
            [pred_encoded]
        )[0]
    )


    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    probabilities = model.predict_proba(
        user_data
    )[0]

    confidence = float(
        np.max(probabilities)
    )


    # ========================================================
    # SHAP VALUES
    # ========================================================

    shap_output = explainer.shap_values(
        user_data
    )


    # ========================================================
    # HANDLE DIFFERENT SHAP VERSIONS
    # ========================================================

    if hasattr(
        shap_output,
        "values"
    ):

        shap_values = shap_output.values

    else:

        shap_values = shap_output


    # --------------------------------------------------------
    # SHAP OUTPUT CONVERSION
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        # Older SHAP versions
        class_index = int(
            pred_encoded
        )

        if class_index >= len(
            shap_values
        ):

            class_index = 0

        shap_for_class = np.asarray(
            shap_values[class_index][0]
        )


    elif len(
        np.asarray(shap_values).shape
    ) == 3:

        # Newer SHAP:
        # (samples, features, classes)

        class_index = int(
            pred_encoded
        )

        class_index = min(
            class_index,
            shap_values.shape[2] - 1
        )

        shap_for_class = np.asarray(
            shap_values[
                0,
                :,
                class_index
            ]
        )


    else:

        # Binary / simple output

        shap_for_class = np.asarray(
            shap_values[0]
        )


    # ========================================================
    # CREATE FEATURE IMPACT DATA
    # ========================================================

    feature_impact = []

    for i, feature in enumerate(
        features
    ):

        value = float(
            shap_for_class[i]
        )

        feature_impact.append({

            "feature": feature,

            "display_name":
                feature_display_names.get(
                    feature,
                    feature
                ),

            "value": round(
                value,
                4
            ),

            "absolute":
                round(
                    abs(value),
                    4
                ),

            "direction":
                "increase"
                if value > 0
                else "decrease"

        })


    # ========================================================
    # SORT BY ABSOLUTE SHAP VALUE
    # ========================================================

    feature_impact.sort(
        key=lambda x: x["absolute"],
        reverse=True
    )


    # ========================================================
    # TOP SHAP FEATURES
    # ========================================================

    top_features = (
        feature_impact[:4]
    )


    # ========================================================
    # GENERATE REASONS FROM SHAP
    # ========================================================

    reasons = []

    for item in top_features:

        feature = item[
            "display_name"
        ]

        value = item[
            "value"
        ]

        if value > 0:

            reasons.append(
                f"{feature} has a positive "
                f"SHAP contribution of "
                f"{abs(value):.3f}, indicating "
                f"that this factor is increasing "
                f"the model's predicted likelihood "
                f"of {prediction}."
            )

        elif value < 0:

            reasons.append(
                f"{feature} has a negative "
                f"SHAP contribution of "
                f"{abs(value):.3f}, indicating "
                f"that this factor is reducing "
                f"the model's predicted likelihood "
                f"of {prediction}."
            )

        else:

            reasons.append(
                f"{feature} has very little "
                f"influence on the current "
                f"prediction according to "
                f"the SHAP analysis."
            )


    # ========================================================
    # PERSONALIZED RECOMMENDATIONS
    # ========================================================

    tips = []

    recommendation_map = {

        "Sleep Duration":
            "Maintain a consistent sleep schedule and aim for approximately 7–8 hours of quality sleep each night.",

        "Physical Activity Level":
            "Increase regular physical activity through walking, jogging, stretching, or other suitable exercises.",

        "BMI Category":
            "Maintain a healthy body weight through balanced nutrition, regular physical activity, and healthy lifestyle habits.",

        "Systolic":
            "Monitor blood pressure regularly, reduce excessive salt intake, stay physically active, and manage daily stress.",

        "Diastolic":
            "Monitor your blood pressure and maintain healthy sleep, diet, exercise, and stress-management habits.",

        "Age":
            "Maintain healthy lifestyle habits and undergo appropriate periodic health monitoring as health risks can change with age.",

        "Gender":
            "Consider overall lifestyle and health factors rather than relying on gender alone, and maintain regular preventive health practices."

    }


    # --------------------------------------------------------
    # RECOMMENDATIONS BASED ON POSITIVE SHAP FEATURES
    # --------------------------------------------------------

    for item in top_features:

        feature = item["feature"]

        value = item["value"]

        if value > 0:

            if feature in recommendation_map:

                tips.append(
                    recommendation_map[
                        feature
                    ]
                )


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    tips = list(
        dict.fromkeys(tips)
    )


    # --------------------------------------------------------
    # DEFAULT RECOMMENDATIONS
    # --------------------------------------------------------

    if not tips:

        tips = [

            "Continue maintaining a consistent sleep schedule and healthy daily routine.",

            "Stay physically active and maintain a balanced diet.",

            "Monitor important health indicators such as blood pressure and body weight.",

            "Practice good sleep hygiene and manage daily stress effectively."

        ]


    # ========================================================
    # RISK LEVEL
    # ========================================================

    prediction_lower = (
        str(prediction).lower()
    )


    if prediction_lower in [
        "none",
        "normal"
    ]:

        risk = "Low"

        color = "#16a34a"

        score = int(
            confidence * 40
        )


    elif prediction_lower == "insomnia":

        risk = "Medium"

        color = "#f59e0b"

        score = int(
            confidence * 70
        )


    else:

        risk = "High"

        color = "#ef4444"

        score = int(
            confidence * 100
        )


    # ========================================================
    # LIMIT SCORE
    # ========================================================

    score = max(
        0,
        min(
            score,
            100
        )
    )


    # ========================================================
    # SEND DATA TO RESULT.HTML
    # ========================================================

    return render_template(

        "result.html",

        prediction=prediction,

        risk=risk,

        color=color,

        score=score,

        confidence=round(
            confidence * 100,
            2
        ),

        reasons=reasons,

        tips=tips,

        feature_impact=feature_impact

    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    message = data.get("message", "").strip()

    if not message:
        return {
            "reply": "Please enter a question."
        }, 400

    context = data.get("context", "")

    reply = chat_with_groq(
        message=message,
        context=context
    )

    return {
        "reply": reply
    } 

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
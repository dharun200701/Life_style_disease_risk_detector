import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load and preprocess the data
def load_and_preprocess_data():
    # Load the datasets
    sleep_df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    synthetic_df = pd.read_csv('synthetic_health_lifestyle_dataset.csv')

    # Combine the datasets
    df = pd.concat([sleep_df, synthetic_df], ignore_index=True)

    # Drop rows with missing target values
    df.dropna(subset=['Sleep Disorder'], inplace=True)


    # Define features and target
    features = ['Age','Gender','Occupation','Sleep Duration','Physical Activity Level','BMI Category','Systolic','Diastolic']
    target = 'Sleep Disorder'

    # Split Blood Pressure
    if 'Blood Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
        df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
        df.drop(columns=['Blood Pressure'], inplace=True)

    # Label Encoders
    encoders = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Fill NaN values
    for col in features:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, encoders, features, target

model, encoders, features, target = load_and_preprocess_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_data = pd.DataFrame([data])

    # Encode categorical features
    for col in user_data.select_dtypes(include=['object']).columns:
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels
            user_data[col] = user_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Ensure all feature columns are present
    for col in features:
        if col not in user_data.columns:
            user_data[col] = 0 # or some other default value

    user_data = user_data[features]


    prediction_encoded = model.predict(user_data)[0]
    prediction = encoders[target].inverse_transform([prediction_encoded])[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

# 🩺 Life Style Disease Risk Detector

An **AI-powered lifestyle and sleep health risk prediction system** that analyzes lifestyle and health parameters to predict potential sleep-related health conditions. The system combines **Machine Learning, SHAP Explainable AI, and a Groq-powered health assistant** to provide understandable and personalized results.

## 🚀 Live Demo

### 🌐 Try the Application

**https://life-style-disease-risk-detector.onrender.com**

> ⚠️ The application is intended for educational and research purposes. It is not a substitute for professional medical diagnosis or treatment.

---

## 📌 Project Overview

Lifestyle-related health problems are increasingly influenced by factors such as sleep duration, physical activity, BMI, age, gender, and blood pressure.

This project provides an interactive web application where users enter their lifestyle and health information. A **Random Forest machine learning model** analyzes the information and predicts the most likely sleep-related condition.

The system also uses **SHAP (SHapley Additive exPlanations)** to explain which input features contributed most to the prediction.

An integrated **Groq AI Health Assistant** allows users to ask questions about their prediction and receive AI-generated lifestyle guidance.

---

## ✨ Key Features

* 🧠 **Machine Learning Prediction**

  * Random Forest classification model
  * Predicts sleep-related health conditions

* 📊 **Risk Assessment**

  * Calculates prediction confidence
  * Provides a simplified risk score
  * Categorizes results into Low, Medium, and High risk

* 🔍 **SHAP Explainable AI**

  * Shows which health factors influenced the prediction
  * Displays positive and negative feature contributions
  * Makes machine-learning predictions easier to understand

* 🤖 **AI Health Assistant**

  * Powered by Groq
  * Answers questions related to the prediction
  * Provides lifestyle-oriented recommendations
  * Uses the user's prediction context

* 💡 **Personalized Recommendations**

  * Suggestions based on important contributing factors
  * Covers sleep, physical activity, BMI, blood pressure, and lifestyle habits

* 🎨 **Interactive Web Interface**

  * Responsive HTML/CSS/JavaScript frontend
  * Flask-based backend
  * Interactive prediction and results dashboard

* 📥 **Trained Model Download**

  * Allows users to download the trained model

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    │  Health Information  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Flask Web App    │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Processing    │
                    │ Pandas + Encoders    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Random Forest      │
                    │   Classification     │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │  SHAP Analysis  │        │ Risk Assessment │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │    Results Dashboard │
                    │ Prediction + SHAP +  │
                    │ Recommendations      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Groq AI Assistant  │
                    │   Health Q&A         │
                    └──────────────────────┘
```

---

## 🧠 Machine Learning

### Algorithm

The primary prediction model is:

**Random Forest Classifier**

Configuration includes:

* 200 decision trees
* Balanced class weights
* Fixed random state for reproducibility
* Parallel training using available CPU cores

### Input Features

The model uses the following health and lifestyle parameters:

| Feature                 | Description              |
| ----------------------- | ------------------------ |
| Age                     | User's age               |
| Gender                  | User's gender            |
| Sleep Duration          | Average daily sleep      |
| Physical Activity Level | Daily activity level     |
| BMI Category            | Body Mass Index category |
| Systolic BP             | Systolic blood pressure  |
| Diastolic BP            | Diastolic blood pressure |

### Target

The prediction target is:

**Sleep Disorder**

The system can identify sleep-related categories such as:

* None
* Insomnia
* Sleep Apnea

---

## 🔍 Explainable AI with SHAP

A major feature of this project is **SHAP-based explainability**.

Instead of showing only:

```text
Prediction: Insomnia
```

the application also explains **why the model made that prediction**.

For example:

```text
Sleep Duration       ██████████
Physical Activity    ███████
BMI Category         █████
Blood Pressure       ███
```

The SHAP analysis identifies the relative contribution of each feature to the prediction.

This improves transparency and helps users understand the factors associated with the model's output.

---

## 🤖 Groq AI Health Assistant

The application includes an AI-powered health assistant using the **Groq API**.

Users can ask questions such as:

```text
Why did I get this prediction?

What lifestyle changes should I make?

Explain my SHAP factors.

How can I improve my sleep?
```

The assistant receives the prediction context and provides general lifestyle-oriented information.

The API key is stored securely as an environment variable:

```text
GROQ_API_KEY
```

It is **not stored in the source code**.

---

## 📁 Project Structure

```text
Life_style_disease_risk_detector/
│
├── app.py
├── groq_chatbot.py
├── lifestyle.py
├── model_comparsion.py
├── preprocessing.py
│
├── model_metadata.pkl
├── sleep_disorder_model.pkl
│
├── Sleep_health_and_lifestyle_dataset.csv
├── synthetic_health_lifestyle_dataset.csv
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   ├── script.js
│   ├── result.css
│   └── result.js
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript
* Jinja2

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* Random Forest
* Pandas
* NumPy

### Explainable AI

* SHAP

### Generative AI

* Groq API

### Deployment

* Render
* Gunicorn

### Development

* Git
* GitHub
* VS Code

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/dharun200701/Life_style_disease_risk_detector.git
```

### 2. Move into the project directory

```bash
cd Life_style_disease_risk_detector
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API

Create a `.env` file locally:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

### 6. Run locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## ☁️ Deployment

The application is deployed using **Render** as a Flask web service.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

### Environment Variable

```text
GROQ_API_KEY=your_groq_api_key
```

### Live Application

🌐 **https://life-style-disease-risk-detector.onrender.com**

---

## 🔄 Application Workflow

```text
1. User opens the web application
            ↓
2. Enters lifestyle and health information
            ↓
3. Flask receives the submitted data
            ↓
4. Data is preprocessed
            ↓
5. Random Forest generates prediction
            ↓
6. Prediction confidence is calculated
            ↓
7. SHAP calculates feature contributions
            ↓
8. Risk level and score are generated
            ↓
9. Personalized recommendations are created
            ↓
10. Results dashboard is displayed
            ↓
11. User can interact with the Groq AI Assistant
```

---

## 📊 Explainability

The project focuses not only on prediction but also on **understanding the prediction**.

### Traditional ML

```text
Input → Model → Prediction
```

### This Project

```text
Input
  ↓
Machine Learning Model
  ↓
Prediction
  ↓
SHAP Explanation
  ↓
Risk Assessment
  ↓
Personalized Recommendations
  ↓
AI Health Assistant
```

This makes the system more suitable for demonstrating **Explainable AI (XAI)** concepts.

---

## 🎯 Objectives

* Predict lifestyle-related sleep health risks using machine learning.
* Analyze important health and lifestyle factors.
* Provide interpretable ML predictions using SHAP.
* Generate personalized lifestyle recommendations.
* Integrate generative AI for interactive health-related assistance.
* Build and deploy a complete AI-powered healthcare web application.

---

## 🔮 Future Enhancements

* [ ] Add more lifestyle disease prediction models
* [ ] Improve model validation and cross-validation
* [ ] Add model performance comparison dashboard
* [ ] Add prediction history
* [ ] Add user authentication
* [ ] Add database support
* [ ] Add interactive health analytics
* [ ] Improve mobile responsiveness
* [ ] Add multilingual support
* [ ] Containerize the application using Docker
* [ ] Optimize model loading for faster deployment
* [ ] Add automated model monitoring

---

## ⚠️ Disclaimer

This application is developed for **educational, research, and demonstration purposes**.

The predictions and recommendations generated by this system should **not be considered medical advice, diagnosis, or treatment**.

Users should consult a qualified healthcare professional for medical concerns.

---

## 👨‍💻 Developer

**Dharun**

GitHub:
https://github.com/dharun200701

Project Repository:
https://github.com/dharun200701/Life_style_disease_risk_detector

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Live Demo:**
https://life-style-disease-risk-detector.onrender.com

---

### 📌 Keywords

`Machine Learning` · `Random Forest` · `Flask` · `SHAP` · `Explainable AI` · `Healthcare AI` · `Lifestyle Disease` · `Sleep Disorder Prediction` · `Groq AI` · `Python` · `Artificial Intelligence` · `Web Application` · `Render`

```
```

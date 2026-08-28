# Lifestyle Disease Risk Prediction System

An AI-powered web application that predicts the risk of lifestyle-related diseases based on user-provided health and lifestyle information. The system combines **Machine Learning**, **web technologies**, and an **AI chatbot** to provide users with risk predictions, health-related insights, and an interactive experience.

---

## 📌 Project Overview

Lifestyle-related diseases such as **diabetes, heart disease, hypertension, and obesity** are strongly associated with factors such as diet, physical activity, sleep, stress, age, and other health conditions.

This project aims to provide an accessible early-risk assessment system by analyzing user input through a trained machine learning model.

The application allows users to:

* Enter personal and lifestyle information.
* Submit the information through a web interface.
* Predict the user's disease-risk level using a trained ML model.
* Display the prediction through a dedicated result page.
* Provide understandable health-related information.
* Interact with an AI chatbot for additional guidance.
* Receive a simple and user-friendly interpretation of the prediction.

> **Disclaimer:** This application is intended for educational and preliminary risk-assessment purposes only. It is not a substitute for professional medical diagnosis or treatment.

---

## 🎯 Objectives

The major objectives of the project are:

1. To develop a machine-learning-based lifestyle disease risk prediction system.
2. To analyze important lifestyle and health-related factors.
3. To provide an easy-to-use web interface for prediction.
4. To present prediction results in a clear and understandable format.
5. To integrate an AI-powered chatbot for interactive assistance.
6. To encourage users to become more aware of lifestyle factors associated with disease risk.
7. To provide a foundation for future healthcare-oriented AI applications.

---

## ✨ Key Features

### 1. User Input Module

Users can enter relevant information such as:

* Age
* Gender
* Height and weight
* BMI-related information
* Physical activity
* Dietary habits
* Sleep patterns
* Smoking/alcohol-related lifestyle factors
* Existing health-related information
* Other model-specific parameters

The collected values are processed before being provided to the prediction model.

### 2. Machine Learning Prediction

The system processes the submitted information and uses a trained machine learning model to estimate the user's disease-risk category.

The prediction pipeline generally consists of:

```text
User Input
     ↓
Data Validation
     ↓
Preprocessing
     ↓
Feature Transformation
     ↓
Trained ML Model
     ↓
Risk Prediction
     ↓
Result Page
```

### 3. Result Page

The result page presents the prediction in a simple format so that users can easily understand the outcome.

It can display:

* Predicted risk
* Risk category
* Relevant health information
* Recommendations
* Additional guidance

### 4. AI Chatbot

The project includes an AI chatbot integrated using **Groq API**.

The chatbot provides an interactive way for users to ask questions related to:

* Lifestyle
* Healthy habits
* Exercise
* Nutrition
* Sleep
* General health awareness
* Understanding prediction results

The chatbot is designed as an additional assistance feature and does not replace professional medical advice.

### 5. Responsive Web Interface

The frontend is designed to provide a simple and accessible experience across different screen sizes.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │       User           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Web Interface      │
                    │ HTML / CSS / JS      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Input Validation &  │
                    │    Preprocessing     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   ML Prediction      │
                    │       Model          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Result Page       │
                    │  Risk Interpretation │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   AI Health Chatbot  │
                    │     Groq API         │
                    └──────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

The machine learning workflow includes the following stages:

### Data Collection

A suitable lifestyle/health dataset is used to train the prediction model.

### Data Preprocessing

The dataset is cleaned and transformed before training.

Typical preprocessing operations include:

* Handling missing values
* Removing unnecessary attributes
* Encoding categorical variables
* Feature scaling/normalization
* Selecting relevant features
* Splitting data into training and testing sets

### Model Training

The processed dataset is used to train the selected machine learning algorithm.

### Model Evaluation

The model can be evaluated using metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Model Deployment

The trained model is saved and integrated with the web application so that new user inputs can be processed and predictions can be generated.

---

## 🤖 AI Chatbot Architecture

The chatbot uses the Groq API to generate conversational responses.

```text
User
  │
  ▼
Chatbot Interface
  │
  ▼
JavaScript Request
  │
  ▼
Backend / API Layer
  │
  ▼
Groq API
  │
  ▼
AI-generated Response
  │
  ▼
Chatbot Interface
```

The chatbot operates independently from the ML prediction pipeline while providing additional conversational assistance.

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* JavaScript
* Responsive UI design

### Backend

* Python
* Flask

### Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy
* Joblib/Pickle for model persistence

### AI Integration

* Groq API
* Large Language Model based chatbot

### Development Tools

* Git
* GitHub
* Visual Studio Code
* Python Virtual Environment

---

## 📂 Project Structure

The project is organized approximately as follows:

```text
Lifestyle-Disease-Prediction/
│
├── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── script.js
│   │
│   └── images/
│
├── model/
│   └── trained_model.pkl
│
├── dataset/
│   └── dataset.csv
│
└── notebooks/
    └── model_training.ipynb
```

> The exact file/folder names may vary depending on the latest project version.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

### 2. Navigate to the Project Directory

```bash
cd Lifestyle-Disease-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

The Groq API key should **not** be hard-coded in the source code.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Make sure `.env` is included in `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

Never commit your actual API key to GitHub.

---

## ▶️ Running the Application

After activating the virtual environment and installing the dependencies:

```bash
python app.py
```

The Flask development server will start.

Open the local application in your browser using the address displayed by Flask, typically:

```text
http://127.0.0.1:5000/
```

---

## 🔄 Application Workflow

```text
1. User opens the application
          ↓
2. User enters health/lifestyle information
          ↓
3. Input validation is performed
          ↓
4. Data is preprocessed
          ↓
5. ML model receives the processed features
          ↓
6. Risk prediction is generated
          ↓
7. Result is displayed
          ↓
8. User can interact with the AI chatbot
```

---

## 📊 Prediction Module

The prediction module is responsible for converting user-provided information into the format expected by the trained model.

The general process is:

```python
Input Data
    ↓
Feature Extraction
    ↓
Encoding
    ↓
Scaling
    ↓
Model Prediction
    ↓
Risk Result
```

The trained model is loaded by the backend and used to generate predictions for new user inputs.

---

## 💬 Chatbot Module

The chatbot provides a conversational interface on the result/application page.

Users can ask questions such as:

```text
What lifestyle changes can reduce disease risk?
```

```text
Why is physical activity important?
```

```text
How can I improve my sleep?
```

The chatbot sends the user's query to the configured AI service and displays the generated response.

---

## 🗄️ Data Storage

The application does not necessarily require a traditional database for the basic prediction workflow.

The system primarily uses:

* Dataset files for model training.
* A saved trained model for prediction.
* User input temporarily during the prediction request.
* Environment variables for sensitive API configuration.

If a database is added in a future version, it could be used to store user profiles, prediction history, chatbot conversations, or analytics.

---

## 🔒 Security Considerations

The following security practices should be followed:

* Never expose API keys in frontend JavaScript.
* Store secrets in environment variables.
* Do not commit `.env` files.
* Validate user inputs on the backend.
* Avoid storing sensitive health information unnecessarily.
* Use HTTPS when deploying the application publicly.
* Keep dependencies updated.

---

## 🧪 Testing

The application can be tested at multiple levels.

### Functional Testing

Verify:

* Input form submission
* Prediction generation
* Result-page rendering
* Chatbot interaction
* Error handling

### Model Testing

Evaluate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### UI Testing

Check the application on:

* Desktop
* Tablet
* Mobile devices
* Different modern browsers

---

## 🚀 Future Enhancements

Potential improvements include:

* Support for multiple disease prediction models.
* Personalized lifestyle recommendations.
* User authentication.
* Prediction history.
* Database integration.
* Interactive health dashboards.
* Improved explainability using feature importance/SHAP.
* Mobile application development.
* Multilingual chatbot support.
* Cloud deployment.
* Integration with wearable-device data.
* More advanced deep-learning models.
* Continuous model improvement using validated datasets.

---

## 🌍 Sustainable Development Goals

The project is related to the following **United Nations Sustainable Development Goals (SDGs)**:

### SDG 3 — Good Health and Well-being

The project supports health awareness and early risk assessment by using AI and machine learning to analyze lifestyle-related risk factors.

### SDG 9 — Industry, Innovation and Infrastructure

The project demonstrates the application of artificial intelligence and machine learning in a healthcare-oriented system.

### SDG 10 — Reduced Inequalities

A web-based health-risk awareness system can potentially make preliminary health information more accessible to a wider range of users.

---

## ⚠️ Medical Disclaimer

This project is an **academic and educational prototype**.

The predictions generated by this application should not be considered a medical diagnosis. Users should consult qualified healthcare professionals for medical evaluation, diagnosis, treatment, and personalized health advice.

---

## 👨‍💻 Contributors

**Lifestyle Disease Risk Prediction Project**

Developed as an academic project focused on applying:

* Machine Learning
* Artificial Intelligence
* Web Development
* Healthcare Technology

---

## 📜 License

This project is intended primarily for academic and educational purposes.

If you choose to publish it as open source, add an appropriate license such as the MIT License.

---

## ⭐ Acknowledgements

The project makes use of open-source technologies and machine-learning libraries, including Python, Flask, Pandas, NumPy, Scikit-learn, and related tools.

Special acknowledgement is given to the datasets, research papers, open-source libraries, and AI technologies used during the development and evaluation of the project.

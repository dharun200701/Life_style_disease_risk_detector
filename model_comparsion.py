import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Load datasets
sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
synthetic_df = pd.read_csv("synthetic_health_lifestyle_dataset.csv")

# Combine datasets
df = pd.concat([sleep_df, synthetic_df], ignore_index=True)

# Remove rows without target
df.dropna(subset=["Sleep Disorder"], inplace=True)


# Split Blood Pressure column
if "Blood Pressure" in df.columns:
    df[["Systolic","Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
    df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)


# Features and target
features = [
    "Age","Gender","Sleep Duration","Physical Activity Level",
    "BMI Category","Daily Steps","Systolic","Diastolic",
    "Alcohol_Consumption","Smoker"
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
    if df[col].dtype != "object":
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


# Split dataset
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}


# Store accuracies
model_names = []
accuracies = []

print("\nModel Accuracy Comparison\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {accuracy:.4f}")

    model_names.append(name)
    accuracies.append(accuracy)


# Visualization
plt.figure(figsize=(8,5))
plt.bar(model_names, accuracies)
plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.show()
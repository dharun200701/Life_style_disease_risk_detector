import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Function to show null values
# -----------------------------
def show_null_values(df, dataset_name):
    print(f"\n===== NULL VALUES IN {dataset_name} =====")
    null_counts = df.isnull().sum()
    print(null_counts)
    print("Total Null Values:", null_counts.sum())


# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_data(df, target_column, drop_columns=None):
    df = df.copy()

    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Convert numeric-like columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Handle missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y


# =====================================
# DATASET 1: Sleep Health & Lifestyle
# =====================================
df1 = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

show_null_values(df1, "DATASET 1")

X1, y1 = preprocess_data(
    df1,
    target_column="Sleep Disorder",
    drop_columns=["Person ID"]
)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = RandomForestClassifier(random_state=42)

model1.fit(X_train1, y_train1)

y_pred1 = model1.predict(X_test1)

print("\n===== DATASET 1 RESULTS =====")
print("Accuracy:", accuracy_score(y_test1, y_pred1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test1, y_pred1))
print("\nClassification Report:\n", classification_report(y_test1, y_pred1))


# =====================================
# DATASET 2: Synthetic Health Dataset
# =====================================
df2 = pd.read_csv("synthetic_health_lifestyle_dataset.csv")

show_null_values(df2, "DATASET 2")

X2, y2 = preprocess_data(
    df2,
    target_column="Chronic_Disease",
    drop_columns=["ID"]
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = RandomForestClassifier(random_state=42)
model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)

print("\n===== DATASET 2 RESULTS =====")
print("Accuracy:", accuracy_score(y_test2, y_pred2))
print("\nConfusion Matrix:\n", confusion_matrix(y_test2, y_pred2))
print("\nClassification Report:\n", classification_report(y_test2, y_pred2))

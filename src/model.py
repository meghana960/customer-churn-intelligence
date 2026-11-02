"""Training and inference utilities for the churn demo."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["TenureMonths", "MonthlyCharges"]
CATEGORICAL_FEATURES = [
    "Contract", "TechSupport", "InternetService", "PaymentMethod"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def train_model(data):
    X = data[FEATURES]
    y = data["Churn"]
    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=180, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return {"pipeline": pipeline, "metrics": metrics}

def predict_customer(pipeline, customer):
    import pandas as pd
    row = pd.DataFrame([customer], columns=FEATURES)
    return float(pipeline.predict_proba(row)[0, 1])

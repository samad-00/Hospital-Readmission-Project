import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_prior_visits"
]

CATEGORICAL_FEATURES = [
    "age",
    "gender",
    "race",
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "department",
    "primary_diagnosis",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change",
    "diabetesMed"
]

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def get_classifier(model_name="Decision Tree", **kwargs):
    if model_name == "Decision Tree":
        max_depth = kwargs.get("max_depth", 6)
        min_samples_split = kwargs.get("min_samples_split", 20)
        class_weight = kwargs.get("class_weight", "balanced")
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42
        )
    elif model_name == "Random Forest":
        n_estimators = kwargs.get("n_estimators", 50)
        max_depth = kwargs.get("max_depth", 8)
        class_weight = kwargs.get("class_weight", "balanced")
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "Logistic Regression":
        C = kwargs.get("C", 1.0)
        class_weight = kwargs.get("class_weight", "balanced")
        return LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def train_and_evaluate_model(
    df,
    model_name="Decision Tree",
    test_size=0.2,
    threshold=0.45,
    **kwargs
):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    clf = get_classifier(model_name, **kwargs)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipeline.fit(X_train, y_train)

    # Probabilities
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Predictions based on tuned threshold
    y_pred_test = (y_prob_test >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_prob_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC & PR curves data
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob_test)
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_test, y_prob_test)

    # Feature Importance (if applicable)
    feature_importances = []
    if hasattr(clf, "feature_importances_"):
        ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
        all_features = NUMERIC_FEATURES + list(ohe_cols)
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)
        feature_importances = fi_df.to_dict(orient="records")
    elif hasattr(clf, "coef_"):
        ohe_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
        all_features = NUMERIC_FEATURES + list(ohe_cols)
        coefs = np.abs(clf.coef_[0])
        fi_df = pd.DataFrame({"Feature": all_features, "Importance": coefs})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)
        feature_importances = fi_df.to_dict(orient="records")

    metrics = {
        "model_name": model_name,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importances": feature_importances
    }

    curve_data = {
        "fpr": fpr[::max(1, len(fpr)//100)].tolist(),
        "tpr": tpr[::max(1, len(tpr)//100)].tolist(),
        "recall_curve": recall_curve[::max(1, len(recall_curve)//100)].tolist(),
        "precision_curve": precision_curve[::max(1, len(precision_curve)//100)].tolist()
    }

    return pipeline, metrics, curve_data, (X_test, y_test, y_prob_test)

def train_and_save_all(clean_parquet_path="data/processed/diabetic_data_clean.parquet"):
    os.makedirs("models", exist_ok=True)
    print(f"Loading {clean_parquet_path}...")
    df = pd.read_parquet(clean_parquet_path)

    models_to_train = [
        ("Decision Tree", {"max_depth": 6, "min_samples_split": 20, "class_weight": "balanced"}),
        ("Random Forest", {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced"}),
        ("Logistic Regression", {"C": 1.0, "class_weight": "balanced"})
    ]

    all_metrics = {}

    for name, params in models_to_train:
        print(f"Training {name}...")
        pipeline, metrics, curves, (X_test, y_test, y_prob) = train_and_evaluate_model(
            df, model_name=name, threshold=0.45, **params
        )
        safe_name = name.lower().replace(" ", "_")
        model_path = f"models/{safe_name}.joblib"
        joblib.dump(pipeline, model_path)
        all_metrics[name] = {
            "metrics": metrics,
            "curves": curves
        }
        print(f"  {name} -> ROC-AUC: {metrics['roc_auc']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save a sample test split for interactive threshold slider & batch explorer
    sample_test = pd.concat([X_test.head(1000), y_test.head(1000)], axis=1)
    sample_test["y_prob_dt"] = y_prob[:1000]
    sample_test.to_csv("data/processed/test_sample_1000.csv", index=False)
    print("All models successfully trained and saved!")

if __name__ == "__main__":
    train_and_save_all()

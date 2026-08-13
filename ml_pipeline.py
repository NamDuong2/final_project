"""
Author: Nam Duong
CSE 163 AA

This module handles feature engineering, model training,
and evaluation for predicting severe traffic crashes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def run_rq3_machine_learning(joined_rq3):
    """
    Train and evaluate classifiers (RQ3) to predict crash severity.
    Returns a tuple of (results_df, feature_imp_df).
    """
    print("\n=== RUNNING RQ3 MACHINE LEARNING PIPELINE ===")

    joined_rq3 = joined_rq3.copy()
    joined_rq3["DAY_OF_WEEK"] = pd.to_datetime(
        joined_rq3["CRASH DATE"]
    ).dt.day_name()

    joined_rq3["CRASH HOUR"] = pd.to_datetime(
        joined_rq3["CRASH TIME"], format="%H:%M", errors="coerce"
    ).dt.hour

    categorical_features = [
        "BOROUGH", "DAY_OF_WEEK", "VEHICLE TYPE CODE 1",
        "CONTRIBUTING FACTOR VEHICLE 1"
    ]
    numeric_features = ["CRASH HOUR", "Ozone_Level"]

    X = joined_rq3[categorical_features + numeric_features].copy()
    y = joined_rq3["SEVERE_CRASH"].astype(int)

    # Impute missing values for modeling features
    for col in categorical_features:
        X[col] = X[col].fillna("Unknown").astype(str)
    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())

    if len(X) > 100000:
        print(f" -> Sampling dataset from {len(X)} to 100,000 rows")
        X, _, y, _ = train_test_split(
            X, y, train_size=100000, random_state=42, stratify=y
        )

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scikit-learn Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False),
             categorical_features)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(solver="saga",
                                                  tol=0.01,
                                                  max_iter=200,
                                                  random_state=42,
                                                  n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=50,
                                                max_depth=10,
                                                random_state=42,
                                                n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100,
                                                        max_depth=6,
                                                        random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f" -> Training {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("classifier", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1-Score": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": (
                        round(roc_auc_score(y_test, y_proba), 3)
                        if y_proba is not None
                        else np.nan
                       ),
        })

    results_df = pd.DataFrame(results)
    print("\n=== RQ3 ML PERFORMANCE SUMMARY ===")
    print(results_df.to_string(index=False))

    # Feature Importance Analysis from Best Model (Gradient Boosting)
    gb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", models["Gradient Boosting"])
    ])
    gb_pipeline.fit(X_train, y_train)

    ohe_cols = (gb_pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .get_feature_names_out(categorical_features)
                )
    all_feature_names = numeric_features + list(ohe_cols)
    importances = gb_pipeline.named_steps["classifier"].feature_importances_

    feature_imp_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_imp_df.head(10).to_string(index=False))

    return results_df, feature_imp_df

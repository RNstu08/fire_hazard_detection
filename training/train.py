import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from evaluate import evaluate_model

# Define directories
DATA_DIR = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_preprocessed.csv"))
    return train_df, test_df

def split_features_labels(df):
    X = df.drop(columns=["fire_label"])
    y = df["fire_label"]
    return X, y

if __name__ == "__main__":
    mlflow.set_experiment("fire_hazard_detection")

    train_df, test_df = load_data()
    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    # ✨ Standardize features (important for many models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to disk
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print(f"✅ Scaler saved at {MODEL_DIR}/scaler.joblib")

    # Define available models
    models = {
        "LogisticRegression": train_logistic_regression,
        "RandomForest": train_random_forest,
        "XGBoost": train_xgboost
    }

    best_model = None
    best_metric = 0
    best_model_name = ""

    for model_name, model_func in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model on scaled data
            model = model_func(X_train_scaled, y_train)

            # Evaluate model
            metrics = evaluate_model(model, X_test_scaled, y_test)

            # Log input example and signature
            input_example = pd.DataFrame(X_test_scaled[:5], columns=X_test.columns)
            signature = infer_signature(X_train_scaled, model.predict_proba(X_train_scaled))

            # Log all evaluation metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))

            # ✨ Log model type as a parameter
            mlflow.log_param("model_type", model_name)

            # ✨ If possible, log important hyperparameters (optional: depends on model_func returning models with get_params())
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log the model with MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                input_example=input_example,
                signature=signature
            )

            print(f"✅ {model_name} Metrics: {metrics}")

            # Check if this model is the best
            if metrics['roc_auc'] > best_metric:
                best_metric = metrics['roc_auc']
                best_model = model
                best_model_name = model_name

    # Save the best model to disk
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model_xgboost.joblib"))
    print(f"🏆 Best model ({best_model_name}) saved at {MODEL_DIR}/best_model_xgboost.joblib")

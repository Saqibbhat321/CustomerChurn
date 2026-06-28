import os
import sys
import joblib
import mlflow
import mlflow.sklearn
sys.path.append(os.path.abspath("."))
from sklearn.ensemble import RandomForestClassifier
from src.utils.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_FOREST_CONFIG)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Allow imports from project root
sys.path.append(os.path.abspath("."))

from src.data.preprocess import preprocess_data

mlflow.set_experiment(
    "Customer Churn Prediction"
)

def train_model():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data()
    with mlflow.start_run():
        model = RandomForestClassifier(
            **RANDOM_FOREST_CONFIG
        )

        model.fit(X_train, y_train)
        mlflow.log_params(
            RANDOM_FOREST_CONFIG
            )
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metrics({

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "roc_auc": roc_auc

        })

        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE")
        print("=" * 50)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}")

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))
        mlflow.sklearn.log_model(sk_model=model,name="random_forest_model")

        mlflow.log_artifact(
        MODEL_PATH
        )

        mlflow.log_artifact(
        PREPROCESSOR_PATH
         )
        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            model,
            MODEL_PATH
        )

        joblib.dump(
            preprocessor,
            PREPROCESSOR_PATH
        )

        print("\nArtifacts saved successfully.")
        print(f"Model Location       : {MODEL_PATH}")
        print(f"Preprocessor Location: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    train_model()
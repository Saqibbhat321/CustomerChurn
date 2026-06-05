import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    data_path = os.path.join(
        "data",
        "processed",
        "cleaned_data.csv"
    )

    df = pd.read_csv(data_path)

    return df


def preprocess_data():
    df = load_data()

    # Remove customer ID
    df = df.drop(columns=["customerID"])

    # Target encoding
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(
        preprocessor,
        "artifacts/preprocessor.pkl"
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data()

    print("Preprocessing completed successfully.")
    print(f"Training samples: {X_train.shape}")
    print(f"Testing samples: {X_test.shape}")
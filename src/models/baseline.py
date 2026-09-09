import polars as pl

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


TRAIN_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/train.parquet"
VAL_PATH = "C:/Users/User/Desktop/ML/industrial-mlops/data/processed/validation.parquet"


def load_data():

    train_df = pl.read_parquet(TRAIN_PATH)
    val_df = pl.read_parquet(VAL_PATH)

    target = "median_house_value"

    X_train = train_df.drop(target).to_pandas()
    y_train = train_df[target].to_pandas()

    X_val = val_df.drop(target).to_pandas()
    y_val = val_df[target].to_pandas()

    return X_train, y_train, X_val, y_val


def build_pipeline(X_train):

    categorical_features = [
        "ocean_proximity"
    ]

    numerical_features = [
        column
        for column in X_train.columns
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline
print ("=== BASELINE MODEL ===")


def main():

    print("Loading data...")

    X_train, y_train, X_val, y_val = load_data()

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    print("\nBuilding pipeline...")

    pipeline = build_pipeline(X_train)
    print("\n=== PIPELINE ===")
    print(pipeline)
    print("\n=== PIPELINE PARAMETERS ===")
    print([key for key in pipeline.get_params().keys() if "estimators" in key or "depth" in key])

    # MLflow experiment
    mlflow.set_experiment("california-housing")

    with mlflow.start_run(run_name="random-forest-baseline"):

        # Parameters
        mlflow.log_param(
            "model",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_param(
            "random_state",
            42
        )

        print("Training Random Forest...")

        pipeline.fit(X_train, y_train)

        print("Training complete.")

        print("\nPredicting validation set...")

        predictions = pipeline.predict(X_val)

        # Metrics
        mae = mean_absolute_error(
            y_val,
            predictions
        )

        rmse = mean_squared_error(
            y_val,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_val,
            predictions
        )

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save model to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=["sklearn.compose._column_transformer._RemainderColsList"
    ]
)

        print("\n=== BASELINE RESULTS ===")

        print(f"MAE  : {mae:,.2f}")
        print(f"RMSE : {rmse:,.2f}")
        print(f"R²   : {r2:.4f}")

        print("\nMLflow run completed.")

if __name__ == "__main__":
    main()
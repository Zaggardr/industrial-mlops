import optuna
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from baseline import load_data, build_pipeline


TRACKING_URI = "sqlite:///C:/Users/User/Desktop/ML/industrial-mlops/mlflow.db"


def objective(trial, X_train, y_train, X_val, y_val):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    pipeline = build_pipeline(X_train)

    pipeline.set_params(
        model__n_estimators=n_estimators,
        model__max_depth=max_depth,
        model__min_samples_split=min_samples_split,
        model__min_samples_leaf=min_samples_leaf,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)
    rmse = mean_squared_error(y_val, predictions) ** 0.5

    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        mlflow.log_metric("rmse", rmse)

    return rmse


def main():
    print("Loading data...")

    X_train, y_train, X_val, y_val = load_data()

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("california-housing")

    study = optuna.create_study(direction="minimize")

    with mlflow.start_run(run_name="optuna-random-forest"):
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
            ),
            n_trials=10,
        )

        print("\n=== OPTUNA RESULTS ===")
        print(f"Best RMSE: {study.best_value:,.2f}")
        print("Best parameters:")

        for parameter, value in study.best_params.items():
            print(f"  {parameter}: {value}")

        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_trials", 10)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)

        print("\nRetraining best model...")

        best_pipeline = build_pipeline(X_train)
        best_pipeline.set_params(
            model__n_estimators=study.best_params["n_estimators"],
            model__max_depth=study.best_params["max_depth"],
            model__min_samples_split=study.best_params["min_samples_split"],
            model__min_samples_leaf=study.best_params["min_samples_leaf"],
        )

        best_pipeline.fit(X_train, y_train)

        predictions = best_pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        r2 = r2_score(y_val, predictions)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            skops_trusted_types=[
                "sklearn.compose._column_transformer._RemainderColsList"
            ],
        )

        print("\n=== BEST MODEL RESULTS ===")
        print(f"MAE  : {mae:,.2f}")
        print(f"RMSE : {rmse:,.2f}")
        print(f"R²   : {r2:.4f}")
        print("\nMLflow run completed.")


if __name__ == "__main__":
    main()
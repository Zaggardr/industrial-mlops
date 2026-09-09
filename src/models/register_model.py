import mlflow


TRACKING_URI = "sqlite:///C:/Users/User/Desktop/ML/industrial-mlops/mlflow.db"

RUN_ID = "7c16ad2d2d254aedaec141a95ab79ab3"
MODEL_NAME = "CaliforniaHousingModel"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"runs:/{RUN_ID}/model"

    print("Registering model...")
    print("Model URI:", model_uri)

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n=== MODEL REGISTERED ===")
    print("Name:", registered_model.name)
    print("Version:", registered_model.version)
    print("Status:", registered_model.status)


if __name__ == "__main__":
    main()
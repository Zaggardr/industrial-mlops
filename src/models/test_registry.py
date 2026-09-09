import pandas as pd
import mlflow


TRACKING_URI = "sqlite:///C:/Users/User/Desktop/ML/industrial-mlops/mlflow.db"
MODEL_URI = "models:/CaliforniaHousingModel/1"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Loading registered model...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # Sample input: a house near the San Francisco Bay Area
    sample_house = pd.DataFrame([{
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "rooms_per_household": 880.0 / 126.0,
        "bedrooms_per_room": 129.0 / 880.0,
        "population_per_household": 322.0 / 126.0,
        "ocean_proximity": "NEAR BAY",
    }])

    print("\nRunning inference...")
    prediction = model.predict(sample_house)

    print("\n=== PREDICTION RESULT ===")
    print(f"Predicted House Value: ${prediction[0]:,.2f}")


if __name__ == "__main__":
    main()
from contextlib import asynccontextmanager
import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///C:/Users/User/Desktop/ML/industrial-mlops/mlflow.db",
)
MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/CaliforniaHousingModel/1",
)

model = None


class HouseInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(gt=0)
    total_bedrooms: float = Field(ge=0)
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0)
    ocean_proximity: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading model from MLflow Registry...")

    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully.")
    except Exception as exc:
        model = None
        print(f"WARNING: Model could not be loaded: {exc}")

    yield

    print("API shutting down.")


app = FastAPI(
    title="California Housing Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "California Housing Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict(house: HouseInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    data = house.model_dump()

    # Reproduce the feature engineering used during training.
    data["rooms_per_household"] = (
        data["total_rooms"] / data["households"]
    )

    data["bedrooms_per_room"] = (
        data["total_bedrooms"] / data["total_rooms"]
    )

    data["population_per_household"] = (
        data["population"] / data["households"]
    )

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)

    return {
        "predicted_median_house_value": float(prediction[0]),
        "model_uri": MODEL_URI,
    }
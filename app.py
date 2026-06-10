import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv

load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")

import pymongo
import pandas as pd

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# ==========================================================
# MongoDB Connection
# ==========================================================

client = pymongo.MongoClient(
    mongo_db_url,
    tlsCAFile=ca
)

database = client[
    DATA_INGESTION_DATABASE_NAME
]

collection = database[
    DATA_INGESTION_COLLECTION_NAME
]

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Home Route
# ==========================================================

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# ==========================================================
# Training Route
# ==========================================================

@app.get("/train")
async def train_route():
    try:

        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return {
            "status": "success",
            "message": "Training completed successfully"
        }

    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ==========================================================
# Prediction Route
# ==========================================================

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:

        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Remove target column if uploaded
        if "Result" in df.columns:
            df = df.drop(columns=["Result"])

        # Load preprocessor
        preprocessor = load_object(
            "final_model/preprocessor.pkl"
        )

        # Load model
        final_model = load_object(
            "final_model/model.pkl"
        )

        # Create network model
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        # Generate predictions
        y_pred = network_model.predict(df)

        # Add prediction column
        df["predicted_column"] = y_pred

        # Create output directory
        os.makedirs(
            "prediction_output",
            exist_ok=True
        )

        # Save predictions
        output_file_path = (
            "prediction_output/output.csv"
        )

        df.to_csv(
            output_file_path,
            index=False
        )

        # Prediction summary
        phishing_count = (
            df["predicted_column"] == -1
        ).sum()

        safe_count = (
            df["predicted_column"] == 1
        ).sum()

        return {
            "status": "success",
            "rows_processed": len(df),
            "phishing_urls": int(phishing_count),
            "safe_urls": int(safe_count),
            "output_file": output_file_path,
            "predictions": df.to_dict(
                orient="records"
            )
        }

    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    app_run(
        app,
        host="0.0.0.0",
        port=8000
    )
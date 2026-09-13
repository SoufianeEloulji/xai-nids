from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src import logger
from src.inference.predictor import Predictor
from src.storage.db import init_db, insert_prediction
from serving.schemas import PredictionInput, PredictionOutput
import pandas as pd


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialisation de la base de données...")
    init_db()

    logger.info("Chargement du Predictor...")
    ml_models["predictor"] = Predictor("MLP_Dynamic_Undersampling_v3", "production", "artifacts/")

    yield

    print("Cleaning memory...")
    ml_models.clear()


app = FastAPI(title="MLP Predictor API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: PredictionInput):
    try:
        predictor = ml_models["predictor"]
        raw_input = data.model_dump(by_alias=True)
        df = pd.DataFrame([raw_input])

        prediction = predictor.predict(df)

        insert_prediction(
            class_id=prediction["class_id"],
            label=prediction["Label"],
            confidence=prediction["confidence"],
            input_data=raw_input,
            top_features=prediction["top_features"],
        )

        return prediction

    except Exception as e:
        logger.error(f"Échec de la prédiction : {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


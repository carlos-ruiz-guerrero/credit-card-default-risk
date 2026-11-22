import json
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, Query, Request

from credit_default.api.schemas import (
    BatchIn,
    BatchOut,
    ClientIn,
    ModelInfo,
    PredictionOut,
)
from credit_default.config import MODELS_DIR
from credit_default.features import add_features

logger = logging.getLogger("credit_default.api")

MODEL_FILE = MODELS_DIR / "credit_default_model.joblib"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """El modelo se carga una sola vez al arrancar, no en cada petición."""
    bundle = joblib.load(MODEL_FILE)
    with open(THRESHOLD_FILE) as f:
        threshold_cfg = json.load(f)

    trained = bundle.get("sklearn_version")
    if trained != sklearn.__version__:
        logger.warning(
            "scikit-learn distinto al del entrenamiento (entrenado con %s, ejecutando %s)",
            trained,
            sklearn.__version__,
        )

    app.state.bundle = bundle
    app.state.threshold_cfg = threshold_cfg
    yield


app = FastAPI(
    title="Credit Card Default Risk API",
    description=(
        "Probabilidad de impago de un cliente de tarjeta de crédito. "
        "Proyecto educativo: no debe usarse para decisiones reales de crédito."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def score(request: Request, clients: list[ClientIn], threshold: float | None) -> list[PredictionOut]:
    bundle = request.app.state.bundle
    used = request.app.state.threshold_cfg["threshold"] if threshold is None else threshold

    df = pd.DataFrame([c.model_dump() for c in clients])
    X = add_features(df)[bundle["features"]]
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    return [
        PredictionOut(probability=round(float(p), 4), threshold=used, flagged=bool(p >= used))
        for p in probabilities
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    bundle = request.app.state.bundle
    cfg = request.app.state.threshold_cfg
    return ModelInfo(
        model_name=bundle["model_name"],
        threshold=cfg["threshold"],
        cost_fn=cfg["cost_fn"],
        cost_fp=cfg["cost_fp"],
        n_features=len(bundle["features"]),
        features=bundle["features"],
        sklearn_version_trained=bundle.get("sklearn_version"),
        sklearn_version_runtime=sklearn.__version__,
    )


@app.post("/predict", response_model=PredictionOut)
def predict(
    client: ClientIn,
    request: Request,
    threshold: float | None = Query(None, ge=0, le=1, description="Umbral alternativo (por defecto, el óptimo)"),
) -> PredictionOut:
    return score(request, [client], threshold)[0]


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(
    payload: BatchIn,
    request: Request,
    threshold: float | None = Query(None, ge=0, le=1),
) -> BatchOut:
    return BatchOut(predictions=score(request, payload.clients, threshold))
import pytest
from fastapi.testclient import TestClient

from credit_default.config import MODELS_DIR


@pytest.fixture(scope="module")
def client():
    if not (MODELS_DIR / "credit_default_model.joblib").exists():
        pytest.skip("No existe el modelo entrenado: ejecuta train y threshold primero")
    from credit_default.api.main import app

    with TestClient(app) as c:  # el `with` ejecuta el lifespan y carga el modelo
        yield c


def payload(**overrides) -> dict:
    data = {
        "limit_bal": 140_000, "age": 35, "education": 2, "marriage": 2,
        **{f"pay_{i}": 0 for i in range(1, 7)},
        **{f"bill_amt{i}": 20_000 for i in range(1, 7)},
        **{f"pay_amt{i}": 2_000 for i in range(1, 7)},
    }
    data.update(overrides)
    return data


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert 0 < body["threshold"] < 1
    assert body["n_features"] == len(body["features"])
    assert "sex" not in body["features"]


def test_predict_returns_a_valid_probability(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["flagged"], bool)


def test_delayed_payments_raise_the_risk(client):
    low = client.post("/predict", json=payload()).json()["probability"]
    high = client.post("/predict", json=payload(pay_1=2, pay_2=2, pay_3=2)).json()["probability"]
    assert high > low


def test_threshold_override(client):
    always = client.post("/predict?threshold=0", json=payload()).json()
    never = client.post("/predict?threshold=1", json=payload()).json()
    assert always["flagged"] is True
    assert never["flagged"] is False


@pytest.mark.parametrize(
    "bad",
    [
        {"age": -5},
        {"age": 17},
        {"education": 9},
        {"pay_1": 99},
        {"limit_bal": 0},
        {"pay_amt1": -10},
    ],
)
def test_invalid_input_is_rejected(client, bad):
    assert client.post("/predict", json=payload(**bad)).status_code == 422


def test_missing_field_is_rejected(client):
    body = payload()
    del body["age"]
    assert client.post("/predict", json=body).status_code == 422


def test_sex_field_is_rejected(client):
    assert client.post("/predict", json=payload(sex=1)).status_code == 422


def test_batch_keeps_order_and_size(client):
    clients = [payload(), payload(pay_1=2, pay_2=2, pay_3=2)]
    preds = client.post("/predict/batch", json={"clients": clients}).json()["predictions"]
    assert len(preds) == 2
    assert preds[1]["probability"] > preds[0]["probability"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"clients": []}).status_code == 422
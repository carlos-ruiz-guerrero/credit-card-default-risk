from pydantic import BaseModel, ConfigDict, Field


class ClientIn(BaseModel):
    """Datos de un cliente. `sex` no existe a propósito: el modelo no lo usa y se rechaza si llega."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "limit_bal": 140000,
                "age": 35,
                "education": 2,
                "marriage": 2,
                "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
                "bill_amt1": 20000, "bill_amt2": 20000, "bill_amt3": 20000,
                "bill_amt4": 20000, "bill_amt5": 20000, "bill_amt6": 20000,
                "pay_amt1": 2000, "pay_amt2": 2000, "pay_amt3": 2000,
                "pay_amt4": 2000, "pay_amt5": 2000, "pay_amt6": 2000,
            }
        },
    )

    limit_bal: float = Field(gt=0, description="Límite de crédito")
    age: int = Field(ge=18, le=100)
    education: int = Field(ge=1, le=4, description="1 posgrado, 2 universidad, 3 bachillerato, 4 otros")
    marriage: int = Field(ge=1, le=3, description="1 casado/a, 2 soltero/a, 3 otros")

    # Estado de pago: -2 y -1 = sin deuda, 0 = pago mínimo, 1..8 = meses de retraso (pay_1 = último mes)
    pay_1: int = Field(ge=-2, le=8)
    pay_2: int = Field(ge=-2, le=8)
    pay_3: int = Field(ge=-2, le=8)
    pay_4: int = Field(ge=-2, le=8)
    pay_5: int = Field(ge=-2, le=8)
    pay_6: int = Field(ge=-2, le=8)

    # Facturas (pueden ser negativas, hay casos así en el dataset original)
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    # Importes pagados
    pay_amt1: float = Field(ge=0)
    pay_amt2: float = Field(ge=0)
    pay_amt3: float = Field(ge=0)
    pay_amt4: float = Field(ge=0)
    pay_amt5: float = Field(ge=0)
    pay_amt6: float = Field(ge=0)


class BatchIn(BaseModel):
    clients: list[ClientIn] = Field(min_length=1, max_length=1000)


class PredictionOut(BaseModel):
    probability: float = Field(description="Probabilidad estimada de impago")
    threshold: float = Field(description="Umbral aplicado")
    flagged: bool = Field(description="True si el cliente queda señalado para revisión")


class BatchOut(BaseModel):
    predictions: list[PredictionOut]


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    cost_fn: float
    cost_fp: float
    n_features: int
    features: list[str]
    sklearn_version_trained: str | None
    sklearn_version_runtime: str
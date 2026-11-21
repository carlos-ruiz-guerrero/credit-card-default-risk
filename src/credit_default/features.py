import numpy as np
import pandas as pd

PAY_COLS = [f"pay_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

NEW_FEATURES = [
    "utilization_1",
    "avg_utilization",
    "avg_pay_ratio",
    "months_late",
    "max_delay",
    "debt_trend",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas calculadas fila a fila (no usan estadísticos del dataset, no hay fuga de datos)."""
    df = df.copy()
    limit = df["limit_bal"]

    # Utilización del límite de crédito
    util = df[BILL_COLS].div(limit, axis=0)
    df["utilization_1"] = util["bill_amt1"]
    df["avg_utilization"] = util.mean(axis=1)

    # Ratio de pago: lo pagado en el mes k frente a la factura del mes anterior (k+1).
    # Si no había deuda previa (factura <= 0) se considera pagado por completo (1.0).
    ratios = []
    for k in range(1, 6):
        prev_bill = df[f"bill_amt{k + 1}"]
        safe_bill = prev_bill.where(prev_bill > 0, 1)
        ratio = np.where(prev_bill > 0, df[f"pay_amt{k}"] / safe_bill, 1.0)
        ratios.append(pd.Series(ratio, index=df.index).clip(0, 1))
    df["avg_pay_ratio"] = pd.concat(ratios, axis=1).mean(axis=1)

    # Retrasos
    df["months_late"] = (df[PAY_COLS] >= 1).sum(axis=1)
    df["max_delay"] = df[PAY_COLS].max(axis=1).clip(lower=0)

    # Tendencia de la deuda (último mes frente a hace seis meses), relativa al límite
    df["debt_trend"] = (df["bill_amt1"] - df["bill_amt6"]) / limit

    return df
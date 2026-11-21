import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from credit_default.config import FIGURES_DIR, FINDINGS_DIR, MODELS_DIR, PROCESSED_FILE
from credit_default.features import add_features
from credit_default.train import FEATURES, RANDOM_STATE, build_models

COST_FN = 5  # coste de conceder crédito a quien impaga
COST_FP = 1  # coste de rechazar o vigilar a un buen cliente
THRESHOLDS = np.round(np.arange(0.05, 0.95, 0.01), 2)


def cost_at(y_true, pred, c_fn=COST_FN, c_fp=COST_FP) -> dict:
    y = np.asarray(y_true)
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "fn": fn,
        "fp": fp,
        "coste_total": fn * c_fn + fp * c_fp,
        "coste_por_cliente": round((fn * c_fn + fp * c_fp) / len(y), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
    }


def cost_curve(y_true, proba) -> pd.DataFrame:
    rows = [{"umbral": t, **cost_at(y_true, (proba >= t).astype(int))} for t in THRESHOLDS]
    return pd.DataFrame(rows)


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "credit_default_model.joblib")
    name = bundle["model_name"]

    df = add_features(pd.read_csv(PROCESSED_FILE))
    X, y = df[FEATURES], df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = clone(build_models()[name])

    # 1. Elegir el umbral con predicciones fuera de muestra del entrenamiento
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    train_curve = cost_curve(y_train, oof)
    best_t = float(train_curve.loc[train_curve["coste_total"].idxmin(), "umbral"])

    # 2. Evaluar en prueba, sin haber tocado esos datos para elegir
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    test_curve = cost_curve(y_test, proba)
    test_curve.to_csv(FINDINGS_DIR / "threshold_curve.csv", index=False)

    summary = pd.DataFrame(
        [
            {"escenario": "Sin modelo (aprobar a todos)", **cost_at(y_test, np.zeros(len(y_test), dtype=int))},
            {"escenario": "Umbral 0,50", **cost_at(y_test, (proba >= 0.5).astype(int))},
            {"escenario": f"Umbral óptimo ({best_t:.2f})", **cost_at(y_test, (proba >= best_t).astype(int))},
        ]
    )
    summary.to_csv(FINDINGS_DIR / "threshold_summary.csv", index=False)

    with open(MODELS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": best_t, "cost_fn": COST_FN, "cost_fp": COST_FP, "model": name}, f, indent=2)

    # Gráfico: coste por cliente frente al umbral
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test_curve["umbral"], test_curve["coste_por_cliente"])
    ax.axvline(best_t, color="red", linestyle="--", label=f"Umbral óptimo = {best_t:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Umbral 0,50")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Coste medio por cliente")
    ax.set_title(f"Coste según el umbral (FN={COST_FN}, FP={COST_FP})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_cost_curve.png", dpi=150)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nUmbral elegido (validación cruzada): {best_t:.2f}")


if __name__ == "__main__":
    main()
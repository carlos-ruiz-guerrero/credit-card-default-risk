import json

import numpy as np
import pandas as pd

from credit_default.config import FINDINGS_DIR, MODELS_DIR, PROCESSED_FILE


def group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col, observed=True):
        y, pred = g["default"], g["flagged"]
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append(
            {
                group_col: group,
                "clientes": len(g),
                "tasa_impago_real": round(y.mean(), 4),
                "tasa_señalados": round(pred.mean(), 4),
                "recall": round(tp / (tp + fn), 4) if tp + fn else np.nan,
                "tasa_falsos_positivos": round(fp / (fp + tn), 4) if fp + tn else np.nan,
                "precision": round(tp / (tp + fp), 4) if tp + fp else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    threshold = json.load(open(MODELS_DIR / "threshold.json"))["threshold"]

    pred = pd.read_csv(FINDINGS_DIR / "test_predictions.csv")
    clean = pd.read_csv(PROCESSED_FILE)[["id", "sex", "age"]]
    df = pred.merge(clean, on="id", how="left")
    assert df["sex"].notna().all(), "Hay IDs sin cruzar"

    df["flagged"] = (df["probability"] >= threshold).astype(int)
    df["sexo"] = df["sex"].map({1: "Hombre", 2: "Mujer"})
    df["edad"] = pd.cut(df["age"], bins=[20, 29, 39, 49, 100], labels=["21-29", "30-39", "40-49", "50+"])

    for col, name in [("sexo", "fairness_by_sex"), ("edad", "fairness_by_age")]:
        table = group_metrics(df, col)
        table.to_csv(FINDINGS_DIR / f"{name}.csv", index=False)
        print(f"\n{table.to_string(index=False)}")

    print(f"\nUmbral usado: {threshold:.2f}")


if __name__ == "__main__":
    main()
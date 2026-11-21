import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from credit_default.config import FIGURES_DIR, FINDINGS_DIR, MODELS_DIR, PROCESSED_FILE
from credit_default.features import add_features
from credit_default.train import FEATURES, RANDOM_STATE


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "credit_default_model.joblib")
    model = bundle["model"]

    df = add_features(pd.read_csv(PROCESSED_FILE))
    X, y = df[FEATURES], df["default"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    importance = (
        pd.DataFrame(
            {
                "variable": FEATURES,
                "importancia": result.importances_mean.round(4),
                "desviacion": result.importances_std.round(4),
            }
        )
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(FINDINGS_DIR / "permutation_importance.csv", index=False)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["variable"], top["importancia"], xerr=top["desviacion"])
    ax.set_xlabel("Caída del ROC-AUC al permutar la variable")
    ax.set_title("Importancia por permutación (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)

    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
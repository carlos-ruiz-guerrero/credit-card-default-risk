import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default.config import FINDINGS_DIR, MODELS_DIR, PROCESSED_FILE
from credit_default.features import (
    BILL_COLS,
    NEW_FEATURES,
    PAY_AMT_COLS,
    PAY_COLS,
    add_features,
)

RANDOM_STATE = 42

# Se excluye `sex` a propósito (equidad); se usará más adelante para auditar el modelo
CATEGORICAL = ["education", "marriage"] + PAY_COLS
NUMERIC = ["limit_bal", "age"] + BILL_COLS + PAY_AMT_COLS + NEW_FEATURES
FEATURES = CATEGORICAL + NUMERIC


def build_models() -> dict:
    logistic = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                        ("num", StandardScaler(), NUMERIC),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, random_state=RANDOM_STATE
    )
    return {"Logistic Regression": logistic, "Gradient Boosting": boosting}


def evaluate(name, model, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "model_name": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "cv_roc_auc": round(cv_auc, 4),
    }, proba


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    df = add_features(pd.read_csv(PROCESSED_FILE))
    X, y, ids = df[FEATURES], df["default"], df["id"]

    X_train, X_test, y_train, y_test, _, id_test = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results, probas, models = [], {}, build_models()
    for name, model in models.items():
        metrics, proba = evaluate(name, model, X_train, y_train, X_test, y_test)
        results.append(metrics)
        probas[name] = proba

    results_df = pd.DataFrame(results)
    results_df.to_csv(FINDINGS_DIR / "model_performance.csv", index=False)
    print(results_df.to_string(index=False))

    best_name = results_df.sort_values("roc_auc", ascending=False).iloc[0]["model_name"]
    joblib.dump(
        {
            "model": models[best_name],
            "model_name": best_name,
            "features": FEATURES,
            "sklearn_version": sklearn.__version__,
        },
        MODELS_DIR / "credit_default_model.joblib",
    )

    pd.DataFrame(
        {"id": id_test.values, "default": y_test.values, "probability": probas[best_name]}
    ).to_csv(FINDINGS_DIR / "test_predictions.csv", index=False)

    print(f"\nMejor modelo: {best_name}  (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
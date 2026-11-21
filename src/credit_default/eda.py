import matplotlib

matplotlib.use("Agg")  # sin ventanas: solo guarda archivos
import matplotlib.pyplot as plt
import pandas as pd

from credit_default.config import FIGURES_DIR, FINDINGS_DIR, PROCESSED_FILE

LABELS = {
    "sex": {1: "Hombre", 2: "Mujer"},
    "education": {1: "Posgrado", 2: "Universidad", 3: "Bachillerato", 4: "Otros"},
    "marriage": {1: "Casado/a", 2: "Soltero/a", 3: "Otros"},
}


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Tasa de impago y número de clientes por categoría."""
    out = (
        df.groupby(col, observed=True)["default"]
        .agg(clientes="count", impagos="sum", tasa_impago="mean")
        .reset_index()
    )
    out["tasa_impago"] = out["tasa_impago"].round(4)
    return out


def bar_chart(table: pd.DataFrame, col: str, title: str, filename: str, rotate: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table[col].astype(str), table["tasa_impago"] * 100)
    ax.axhline(table.attrs["global_rate"] * 100, color="red", linestyle="--", label="Media global")
    ax.set_title(title)
    ax.set_ylabel("Tasa de impago (%)")
    ax.legend()
    plt.xticks(rotation=rotate)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_FILE)
    global_rate = df["default"].mean()

    # Segmentos derivados
    df["age_group"] = pd.cut(
        df["age"], bins=[20, 29, 39, 49, 59, 100], labels=["21-29", "30-39", "40-49", "50-59", "60+"]
    )
    df["limit_group"] = pd.qcut(
        df["limit_bal"], q=4, labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"]
    )

    for col, mapping in LABELS.items():
        df[col + "_label"] = df[col].map(mapping)

    segments = {
        "age_group": ("age_group", "Tasa de impago por edad"),
        "limit_group": ("limit_group", "Tasa de impago por límite de crédito"),
        "education_label": ("education_label", "Tasa de impago por nivel educativo"),
        "marriage_label": ("marriage_label", "Tasa de impago por estado civil"),
        "sex_label": ("sex_label", "Tasa de impago por sexo"),
        "pay_1": ("pay_1", "Tasa de impago según el estado de pago del último mes"),
    }

    for name, (col, title) in segments.items():
        table = rate_table(df, col)
        table.attrs["global_rate"] = global_rate
        table.to_csv(FINDINGS_DIR / f"default_by_{name}.csv", index=False)
        bar_chart(table, col, title, f"default_by_{name}.png")
        print(f"\n{title}\n{table.to_string(index=False)}")

    # Correlación de las variables numéricas con el impago
    corr = (
        df.select_dtypes("number")
        .corr()["default"]
        .drop("default")
        .sort_values(key=abs, ascending=False)
        .rename("correlacion")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    corr.to_csv(FINDINGS_DIR / "correlation_with_default.csv", index=False)

    print(f"\nTasa de impago global: {global_rate:.2%}")
    print(f"\nTop 10 correlaciones con el impago:\n{corr.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
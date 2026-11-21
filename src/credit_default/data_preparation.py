import pandas as pd

from credit_default.config import (
    FINDINGS_DIR,
    PROCESSED_DIR,
    PROCESSED_FILE,
    RAW_FILE,
)


def load_raw() -> pd.DataFrame:
    """Lee el Excel original (la fila 0 son etiquetas genéricas, los nombres están en la fila 1)."""
    return pd.read_excel(RAW_FILE, header=1, engine="calamine")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve el dataset limpio y un informe de auditoría con los cambios realizados."""
    df = df.copy()
    audit = []

    # 1. Nombres de columnas: PAY_0 es en realidad PAY_1, y el objetivo se renombra
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "default"})
    df.columns = [c.lower() for c in df.columns]

    # 2. Valores sin documentar en EDUCATION (0, 5, 6) -> 4 ("otros")
    mask = df["education"].isin([0, 5, 6])
    audit.append(("education", "0, 5, 6 -> 4 (otros)", int(mask.sum())))
    df.loc[mask, "education"] = 4

    # 3. Valores sin documentar en MARRIAGE (0) -> 3 ("otros")
    mask = df["marriage"] == 0
    audit.append(("marriage", "0 -> 3 (otros)", int(mask.sum())))
    df.loc[mask, "marriage"] = 3

    # 4. PAY_*: -2 y 0 no están documentados; se conservan y solo se registran
    for col in [f"pay_{i}" for i in range(1, 7)]:
        n = int(df[col].isin([-2, 0]).sum())
        audit.append((col, "-2 y 0 sin documentar (se conservan)", n))

    # 5. Duplicados (ignorando el ID): solo se informa, no se eliminan
    n_dup = int(df.drop(columns="id").duplicated().sum())
    audit.append(("filas", "duplicadas sin contar id (se conservan)", n_dup))

    report = pd.DataFrame(audit, columns=["columna", "regla", "filas_afectadas"])
    return df, report


def validate(df: pd.DataFrame) -> None:
    assert df.shape == (30000, 25), f"Forma inesperada: {df.shape}"
    assert df["id"].is_unique, "Hay IDs duplicados"
    assert df.isna().sum().sum() == 0, "Hay valores nulos"
    assert set(df["default"].unique()) == {0, 1}, "El objetivo no es binario"


def main() -> None:
    df, report = clean(load_raw())
    validate(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_FILE, index=False)
    report.to_csv(FINDINGS_DIR / "data_audit.csv", index=False)

    print(report.to_string(index=False))
    print(f"\nTasa de impago: {df['default'].mean():.2%}")
    print(f"Guardado en: {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
import pandas as pd
import pytest

from credit_default.config import PROCESSED_FILE
from credit_default.data_preparation import clean, validate


def raw_row(id_, default=0, **overrides) -> dict:
    """Una fila con el formato del Excel original (nombres de columna sin limpiar)."""
    row = {
        "ID": id_,
        "LIMIT_BAL": 50_000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "default payment next month": default,
    }
    row.update({f"PAY_{i}": 0 for i in range(2, 7)})
    row.update({f"BILL_AMT{i}": 1_000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 500 for i in range(1, 7)})
    row.update(overrides)
    return row


def make_raw(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def audit_count(report: pd.DataFrame, column: str) -> int:
    return int(report.set_index("columna").loc[column, "filas_afectadas"])


# ---------- Nombres de columnas ----------

def test_columns_are_renamed_and_lowercased():
    df, _ = clean(make_raw(raw_row(1)))
    assert "pay_0" not in df.columns
    assert "pay_1" in df.columns
    assert "default" in df.columns
    assert all(c == c.lower() for c in df.columns)


def test_pay_1_keeps_the_values_of_the_original_pay_0():
    df, _ = clean(make_raw(raw_row(1, PAY_0=2)))
    assert df.loc[0, "pay_1"] == 2


# ---------- Valores sin documentar ----------

@pytest.mark.parametrize(
    "raw, expected",
    [(0, 4), (5, 4), (6, 4), (1, 1), (2, 2), (3, 3), (4, 4)],
)
def test_education_undocumented_values_are_grouped_as_other(raw, expected):
    df, _ = clean(make_raw(raw_row(1, EDUCATION=raw)))
    assert df.loc[0, "education"] == expected


@pytest.mark.parametrize("raw, expected", [(0, 3), (1, 1), (2, 2), (3, 3)])
def test_marriage_undocumented_value_is_grouped_as_other(raw, expected):
    df, _ = clean(make_raw(raw_row(1, MARRIAGE=raw)))
    assert df.loc[0, "marriage"] == expected


def test_audit_report_counts_changed_rows():
    raw = make_raw(
        raw_row(1, EDUCATION=0, MARRIAGE=0),
        raw_row(2, EDUCATION=5),
        raw_row(3, EDUCATION=6),
        raw_row(4, EDUCATION=2),
    )
    _, report = clean(raw)
    assert audit_count(report, "education") == 3
    assert audit_count(report, "marriage") == 1


def test_pay_status_values_are_kept_and_reported():
    raw = make_raw(
        raw_row(1, PAY_2=-2),
        raw_row(2, PAY_2=0),
        raw_row(3, PAY_2=1),
    )
    df, report = clean(raw)
    assert df["pay_2"].tolist() == [-2, 0, 1]
    assert audit_count(report, "pay_2") == 2


# ---------- Duplicados e integridad ----------

def test_duplicates_are_reported_but_not_removed():
    raw = make_raw(raw_row(1), raw_row(2), raw_row(3, AGE=55))  # 1 y 2 son iguales salvo el ID
    df, report = clean(raw)
    assert len(df) == 3
    assert audit_count(report, "filas") == 1


def test_input_dataframe_is_not_modified():
    raw = make_raw(raw_row(1, EDUCATION=0))
    columns_before = list(raw.columns)
    clean(raw)
    assert list(raw.columns) == columns_before
    assert raw.loc[0, "EDUCATION"] == 0


def test_audit_report_has_the_expected_structure():
    _, report = clean(make_raw(raw_row(1)))
    assert list(report.columns) == ["columna", "regla", "filas_afectadas"]
    assert {"education", "marriage", "pay_1", "pay_6", "filas"} <= set(report["columna"])


# ---------- Validación ----------

def test_validate_rejects_an_unexpected_shape():
    df, _ = clean(make_raw(raw_row(1)))
    with pytest.raises(AssertionError):
        validate(df)


@pytest.fixture(scope="module")
def processed_df() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        pytest.skip("No existe el dataset procesado: ejecuta data_preparation primero")
    return pd.read_csv(PROCESSED_FILE)


def test_processed_dataset_passes_validation(processed_df):
    validate(processed_df)


def test_processed_dataset_has_no_undocumented_categories(processed_df):
    assert set(processed_df["education"].unique()) <= {1, 2, 3, 4}
    assert set(processed_df["marriage"].unique()) <= {1, 2, 3}


def test_validate_detects_missing_values(processed_df):
    df = processed_df.copy()
    df.loc[0, "age"] = None
    with pytest.raises(AssertionError):
        validate(df)


def test_validate_detects_a_non_binary_target(processed_df):
    df = processed_df.copy()
    df.loc[0, "default"] = 2
    with pytest.raises(AssertionError):
        validate(df)
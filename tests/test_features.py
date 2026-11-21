import pandas as pd
import pytest

from credit_default.features import NEW_FEATURES, add_features


def make_client(limit_bal=100_000, age=35, pays=None, bills=None, amounts=None) -> pd.DataFrame:
    """Un cliente con valores neutros, que cada test modifica a su gusto."""
    pays = [0] * 6 if pays is None else pays
    bills = [0] * 6 if bills is None else bills
    amounts = [0] * 6 if amounts is None else amounts

    row = {"limit_bal": limit_bal, "age": age}
    row.update({f"pay_{i + 1}": v for i, v in enumerate(pays)})
    row.update({f"bill_amt{i + 1}": v for i, v in enumerate(bills)})
    row.update({f"pay_amt{i + 1}": v for i, v in enumerate(amounts)})
    return pd.DataFrame([row])


def test_all_new_features_are_created():
    df = add_features(make_client())
    for col in NEW_FEATURES:
        assert col in df.columns


def test_input_dataframe_is_not_modified():
    original = make_client(bills=[50_000] * 6)
    columns_before = list(original.columns)
    add_features(original)
    assert list(original.columns) == columns_before


def test_utilization_is_bill_over_limit():
    df = add_features(make_client(limit_bal=100_000, bills=[50_000, 0, 0, 0, 0, 0]))
    assert df.loc[0, "utilization_1"] == pytest.approx(0.5)


def test_avg_utilization_is_mean_over_six_months():
    df = add_features(make_client(limit_bal=100_000, bills=[60_000, 0, 0, 0, 0, 0]))
    assert df.loc[0, "avg_utilization"] == pytest.approx(0.1)


def test_months_late_counts_only_delays():
    df = add_features(make_client(pays=[2, 1, 0, -1, -2, 0]))
    assert df.loc[0, "months_late"] == 2


def test_max_delay_is_never_negative():
    df = add_features(make_client(pays=[-2, -1, -2, -1, -2, -1]))
    assert df.loc[0, "max_delay"] == 0


def test_max_delay_takes_the_largest_delay():
    df = add_features(make_client(pays=[0, 3, 1, 0, 0, 0]))
    assert df.loc[0, "max_delay"] == 3


def test_debt_trend_is_relative_to_the_limit():
    df = add_features(make_client(limit_bal=100_000, bills=[30_000, 0, 0, 0, 0, 10_000]))
    assert df.loc[0, "debt_trend"] == pytest.approx(0.2)


def test_pay_ratio_is_one_when_there_was_no_previous_debt():
    df = add_features(make_client(bills=[0] * 6, amounts=[0] * 6))
    assert df.loc[0, "avg_pay_ratio"] == pytest.approx(1.0)


def test_pay_ratio_half_paid():
    df = add_features(make_client(bills=[10_000] * 6, amounts=[5_000] * 6))
    assert df.loc[0, "avg_pay_ratio"] == pytest.approx(0.5)


def test_pay_ratio_is_capped_at_one_when_overpaying():
    df = add_features(make_client(bills=[10_000] * 6, amounts=[50_000] * 6))
    assert df.loc[0, "avg_pay_ratio"] == pytest.approx(1.0)


def test_pay_ratio_is_zero_when_nothing_is_paid():
    df = add_features(make_client(bills=[10_000] * 6, amounts=[0] * 6))
    assert df.loc[0, "avg_pay_ratio"] == pytest.approx(0.0)


def test_features_are_computed_row_by_row():
    a = make_client(limit_bal=100_000, bills=[50_000, 0, 0, 0, 0, 0])
    b = make_client(limit_bal=100_000, bills=[10_000, 0, 0, 0, 0, 0])
    df = add_features(pd.concat([a, b], ignore_index=True))
    assert df["utilization_1"].tolist() == pytest.approx([0.5, 0.1])
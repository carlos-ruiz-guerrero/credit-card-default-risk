import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_default.config import FINDINGS_DIR, MODELS_DIR, PROCESSED_FILE  # noqa: E402,F401
from credit_default.features import add_features  # noqa: E402,F401


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_finding(name: str) -> pd.DataFrame:
    path = FINDINGS_DIR / name
    if not path.exists():
        st.error(f"Falta el archivo `{name}` en reports/findings.")
        st.stop()
    return load_csv(str(path))


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "credit_default_model.joblib")


@st.cache_data
def load_threshold() -> dict:
    with open(MODELS_DIR / "threshold.json") as f:
        return json.load(f)
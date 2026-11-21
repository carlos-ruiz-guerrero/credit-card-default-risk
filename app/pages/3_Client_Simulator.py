import pandas as pd
import streamlit as st
from common import add_features, load_model, load_threshold

st.title("🧮 Simulador de cliente")

bundle = load_model()
model, features = bundle["model"], bundle["features"]
threshold = load_threshold()["threshold"]

EDU = {1: "Posgrado", 2: "Universidad", 3: "Bachillerato", 4: "Otros"}
MAR = {1: "Casado/a", 2: "Soltero/a", 3: "Otros"}
PAY_OPTIONS = list(range(-2, 9))


def pay_label(v: int) -> str:
    return str(v) if v <= 0 else f"{v} mes(es) de retraso"


with st.form("cliente"):
    c1, c2 = st.columns(2)
    limit_bal = c1.number_input("Límite de crédito", 10_000, 1_000_000, 140_000, step=10_000)
    age = c2.slider("Edad", 21, 79, 35)
    education = c1.selectbox("Educación", list(EDU), format_func=EDU.get, index=1)
    marriage = c2.selectbox("Estado civil", list(MAR), format_func=MAR.get, index=1)

    st.markdown("**Estado de pago** (pay_1 = último mes, pay_6 = hace seis meses)")
    cols = st.columns(6)
    pay = [
        cols[i].selectbox(f"pay_{i + 1}", PAY_OPTIONS, index=2, format_func=pay_label, key=f"pay_{i + 1}")
        for i in range(6)
    ]

    with st.expander("Facturas e importes pagados (últimos 6 meses)"):
        b_cols = st.columns(6)
        bills = [
            b_cols[i].number_input(f"bill_amt{i + 1}", value=20_000, step=1_000, key=f"bill_{i + 1}")
            for i in range(6)
        ]
        a_cols = st.columns(6)
        amounts = [
            a_cols[i].number_input(f"pay_amt{i + 1}", value=2_000, step=500, key=f"amt_{i + 1}")
            for i in range(6)
        ]

    submitted = st.form_submit_button("Calcular riesgo")

if submitted:
    row = {"limit_bal": limit_bal, "age": age, "education": education, "marriage": marriage}
    row.update({f"pay_{i + 1}": v for i, v in enumerate(pay)})
    row.update({f"bill_amt{i + 1}": v for i, v in enumerate(bills)})
    row.update({f"pay_amt{i + 1}": v for i, v in enumerate(amounts)})

    X = add_features(pd.DataFrame([row]))[features]
    proba = float(model.predict_proba(X)[0, 1])

    m1, m2 = st.columns(2)
    m1.metric("Probabilidad de impago", f"{proba:.1%}")
    m2.metric("Umbral de decisión", f"{threshold:.2f}")
    st.progress(min(proba, 1.0))

    if proba >= threshold:
        st.error("Cliente señalado para revisión.")
    else:
        st.success("Cliente dentro del umbral de aprobación.")

st.caption(
    "Demostración educativa. El modelo se entrenó con clientes de Taiwán de 2005 y no debe usarse "
    "para decisiones reales de crédito."
)
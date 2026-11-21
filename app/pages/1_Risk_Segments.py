import streamlit as st
from common import load_finding

st.title("👥 Segmentos de riesgo")

OPTIONS = {
    "Estado de pago del último mes": "default_by_pay_1.csv",
    "Límite de crédito (cuartiles)": "default_by_limit_group.csv",
    "Edad": "default_by_age_group.csv",
    "Nivel educativo": "default_by_education_label.csv",
    "Estado civil": "default_by_marriage_label.csv",
    "Sexo": "default_by_sex_label.csv",
}

choice = st.selectbox("Segmento", list(OPTIONS))
table = load_finding(OPTIONS[choice])
seg = table.columns[0]

st.bar_chart(table.set_index(table[seg].astype(str))["tasa_impago"], y_label="Tasa de impago")
st.dataframe(table, hide_index=True, width="stretch")

small = table[table["clientes"] < 500]
if not small.empty:
    st.warning(
        "Grupos con menos de 500 clientes (sus porcentajes son poco fiables): "
        + ", ".join(small[seg].astype(str))
    )
st.caption("Las tasas por segmento describen, no explican: un grupo puede impagar más por otra variable.")
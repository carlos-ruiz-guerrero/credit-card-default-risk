import streamlit as st
from common import PROCESSED_FILE, load_csv, load_finding, load_threshold

st.set_page_config(page_title="Credit Card Default Risk", page_icon="💳", layout="wide")

st.title("💳 Credit Card Default Risk")
st.caption(
    "Dataset: Yeh, I. (2009). Default of Credit Card Clients. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H"
)

df = load_csv(str(PROCESSED_FILE))
perf = load_finding("model_performance.csv")
summary = load_finding("threshold_summary.csv")
thr = load_threshold()
best = perf.sort_values("roc_auc", ascending=False).iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{len(df):,}")
c2.metric("Tasa de impago", f"{df['default'].mean():.1%}")
c3.metric("ROC-AUC (prueba)", f"{best['roc_auc']:.3f}")
c4.metric("Umbral de decisión", f"{thr['threshold']:.2f}")

st.subheader("Resultado de negocio")
st.dataframe(
    summary[["escenario", "fn", "fp", "coste_total", "precision", "recall"]],
    hide_index=True,
    width="stretch",
)
st.info(
    f"Se asume que un impago no detectado cuesta {thr['cost_fn']} veces más que rechazar o vigilar "
    f"a un buen cliente. Es una suposición, no un dato: puedes cambiarla en la página "
    f"*Model Performance*."
)

st.subheader("Cómo navegar")
st.markdown(
    "- **Risk Segments**: tasa de impago por edad, límite, educación, estado civil y estado de pago.\n"
    "- **Model Performance**: métricas, umbral según coste, importancia de variables y equidad.\n"
    "- **Client Simulator**: probabilidad de impago de un cliente hipotético."
)
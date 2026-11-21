import numpy as np
import pandas as pd
import streamlit as st
from common import load_finding, load_threshold

st.title("🤖 Rendimiento del modelo")

st.subheader("Comparativa de modelos")
st.dataframe(load_finding("model_performance.csv"), hide_index=True, width="stretch")

# ---------- Umbral y coste ----------
st.subheader("Umbral de decisión y coste")
preds = load_finding("test_predictions.csv")
thr = load_threshold()
y = preds["default"].to_numpy()
p = preds["probability"].to_numpy()

ratio = st.slider(
    "Coste de un impago no detectado frente a un buen cliente señalado por error",
    1, 15, int(thr["cost_fn"]),
)


def evaluate(t: float) -> dict:
    flagged = p >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return {
        "umbral": t,
        "fn": fn,
        "fp": fp,
        "coste_por_cliente": (fn * ratio + fp) / len(y),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


curve = pd.DataFrame([evaluate(float(t)) for t in np.arange(0.05, 0.95, 0.01)])
best = curve.loc[curve["coste_por_cliente"].idxmin()]

chosen = st.slider("Umbral", 0.05, 0.94, float(thr["threshold"]), 0.01)
res = evaluate(chosen)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recall", f"{res['recall']:.1%}")
c2.metric("Precisión", f"{res['precision']:.1%}")
c3.metric("Impagos no detectados", f"{res['fn']:,}")
c4.metric("Buenos clientes señalados", f"{res['fp']:,}")

st.line_chart(curve.set_index("umbral")["coste_por_cliente"], y_label="Coste medio por cliente")
st.caption(
    f"Con esta relación de costes, el mínimo en el conjunto de prueba está en {best['umbral']:.2f}. "
    f"Aquí solo se explora; el umbral oficial ({thr['threshold']:.2f}) se eligió con validación "
    "cruzada sobre entrenamiento, sin usar estos datos."
)

# ---------- Importancia ----------
st.subheader("Variables más importantes")
imp = load_finding("permutation_importance.csv").head(15)
st.bar_chart(imp, x="variable", y="importancia", sort="-importancia", horizontal=True)
st.caption(
    "Importancia por permutación. Variables muy correlacionadas (por ejemplo pay_1 y months_late) "
    "se reparten la importancia, y no implica causalidad."
)

# ---------- Equidad ----------
st.subheader("Auditoría de equidad")
tab_sex, tab_age = st.tabs(["Por sexo", "Por edad"])
with tab_sex:
    st.dataframe(load_finding("fairness_by_sex.csv"), hide_index=True, width="stretch")
    st.caption("El sexo no es una variable del modelo. Las diferencias son pequeñas y proporcionales a la tasa real.")
with tab_age:
    st.dataframe(load_finding("fairness_by_age.csv"), hide_index=True, width="stretch")
    st.caption("La edad sí es una variable del modelo: el grupo 50+ tiene más falsos positivos.")
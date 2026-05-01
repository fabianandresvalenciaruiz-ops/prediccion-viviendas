
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predicción Viviendas", layout="wide")

st.title("🏠 Sistema Inteligente de Predicción de Viviendas")
st.markdown("Estimación de precios usando Machine Learning")

# =========================
# CARGAR MODELO
# =========================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_entrenado.pkl")
    columnas = joblib.load("columnas.pkl")
    return modelo, columnas

modelo, columnas = cargar_modelo()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📥 Ingresar datos")

entrada = {}

for col in columnas[:15]:
    entrada[col] = st.sidebar.number_input(
        col,
        min_value=0.0,
        value=0.0,
        step=1.0
    )

# =========================
# PREDICCIÓN
# =========================
if st.sidebar.button("🔮 Predecir"):

    df = pd.DataFrame([entrada])
    df = df.reindex(columns=columnas, fill_value=0)

    pred = modelo.predict(df)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Precio estimado", f"${pred:,.0f}")

    with col2:
        st.success("Predicción generada correctamente")

# =========================
# DASHBOARD
# =========================
st.subheader("📊 Variables más influyentes")

if hasattr(modelo, "feature_importances_"):

    importancias = modelo.feature_importances_

    if len(importancias) == len(columnas):

        df_imp = pd.DataFrame({
            "Variable": columnas,
            "Importancia": importancias
        }).sort_values(by="Importancia", ascending=False).head(10)

        st.bar_chart(df_imp.set_index("Variable"))

    else:
        st.warning("⚠️ Error en dimensiones de variables")

else:
    st.info("ℹ️ Modelo sin importancia de variables")

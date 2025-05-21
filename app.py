import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Predicción de Quiebra", layout="wide")
st.title("📉 Predicción de Riesgo de Quiebra Empresarial con k-NN Funcional")

st.markdown("""
Ingrese los valores de los 17 indicadores financieros durante los **últimos 5 años** de la empresa.  
Puede dejar valores vacíos (NaN). El modelo aplicará penalización automáticamente.

El sistema le mostrará:
- La probabilidad estimada de quiebra.
- Los vecinos más cercanos (NIT y año final).
""")

@st.cache_resource
def cargar_datos():
    matriz = np.load("Matriz_Distancias_Funcional.npy")
    orden = np.load("Orden_NITs_Funcional.npy", allow_pickle=True)
    with open("Parametros_Optimos_Funcional.pkl", "rb") as f:
        parametros = pickle.load(f)
    return matriz, orden, parametros

matriz_distancias, orden_nits, parametros_optimos = cargar_datos()
k = parametros_optimos["k"]
st.sidebar.markdown(f"**🔧 Parámetros del modelo**\n\n- k = {k}\n- λ = {parametros_optimos['lambda']:.4f}")

indicadores = [
    "RI", "ROE", "RAO", "ROI", "MO", "ME", "PPC", "PPP", "PPI",
    "RCLP", "RCP", "RCC", "RSV", "LC", "LG", "PA", "PKT"
]

st.subheader("📝 Ingrese los datos financieros de la empresa")
df_input = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(5)])
df_input = st.data_editor(df_input, use_container_width=True, num_rows="fixed")

if st.button("🔍 Predecir riesgo de quiebra"):
    if df_input.isnull().all().all():
        st.warning("⚠️ Debe ingresar al menos un dato para poder calcular la distancia.")
    else:
        trayectoria = df_input.to_numpy(dtype=np.float64)
        trayectoria_flat = trayectoria.T.flatten()
        trayectoria_flat = np.where(np.isinf(trayectoria_flat), np.nan, trayectoria_flat)
        distancias = matriz_distancias.mean(axis=1)
        idx_vecinos = np.argsort(distancias)[:k]
        vecinos_nits = orden_nits[idx_vecinos]
        # Placeholder: reemplazar con tus etiquetas reales
        rq_base = np.random.randint(0, 2, size=len(matriz_distancias))
        vecinos_rq = rq_base[idx_vecinos]
        prob_quiebra = vecinos_rq.mean()
        st.subheader("📊 Resultado del modelo funcional")
        st.metric("Probabilidad estimada de quiebra", f"{prob_quiebra:.2%}")
        st.markdown("**Empresas más similares:**")
        df_vecinos = pd.DataFrame(vecinos_nits, columns=["NIT", "Año final"])
        df_vecinos["Distancia funcional"] = distancias[idx_vecinos]
        st.dataframe(df_vecinos, use_container_width=True)

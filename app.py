# --- Streamlit app comparativa para modelo funcional base vs extendido ---
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Configuración general ---
st.set_page_config(page_title="Predicción de Quiebra", layout="wide")
st.title("📉 Comparación del Modelo Funcional k-NN: Base vs Extendido")

# --- Explicación ---
st.markdown("""
Esta herramienta compara dos versiones del modelo funcional k-NN:
- ✅ **Modelo base**: usa solo trayectorias de 17 indicadores financieros.
- ✨ **Modelo extendido**: incluye además el departamento, el sector económico y el año final.

Ingrese los valores de los 17 indicadores para los últimos 5 años. Puede dejar valores faltantes (NaN). Seleccione también el Departamento y el CIIU para el modelo extendido.
""")

# --- Cargar datos y parámetros ---
@st.cache_data
def cargar_datos():
    base_b, params_b = pd.read_parquet("9_1EspacioF.parquet"), pickle.load(open("9_2ParametrosFuncional.pkl", "rb"))
    base_e, params_e = pd.read_parquet("20_1EspacioF.parquet"), pickle.load(open("20_2ParametrosFuncional.pkl", "rb"))
    return base_b, params_b, base_e, params_e

espacioB, paramsB, espacioE, paramsE = cargar_datos()

# --- Extraer info de modelo base ---
k_b = paramsB['k']
lambda_b = paramsB['lambda']
pesos_b = paramsB['pesos']
columnas_b = [col for col in espacioB.columns if "_-" in col]
indicadores = sorted(set(col.split("_-")[0] for col in columnas_b))
n_ventana = len(set(col.split("_-")[1] for col in columnas_b))

# --- Extraer info de modelo extendido ---
k_e = paramsE['k']
lambda_e = paramsE['lambda']
pesos_e = paramsE['pesos']

# --- Inicializar entrada y metadatos ---
if "df_input" not in st.session_state:
    st.session_state.df_input = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(n_ventana)])
if "nit_origen" not in st.session_state:
    st.session_state.nit_origen = None
if "anio_final_usuario" not in st.session_state:
    st.session_state.anio_final_usuario = 2023

# --- Panel lateral con selectores ---
st.sidebar.subheader("📌 Variables cualitativas (modelo extendido)")
opciones_dep = sorted(espacioE["DEP"].dropna().unique())
opciones_ciiu = sorted(espacioE["CIIU_Letra"].dropna().unique())
dep_usuario = st.sidebar.selectbox("Departamento", opciones_dep)
ciiu_usuario = st.sidebar.selectbox("Letra CIIU", opciones_ciiu)

# --- Botón para cargar ejemplo real ---
if st.sidebar.button("🎯 Usar trayectoria real de ejemplo"):
    fila = espacioE.sample(1)
    st.session_state.nit_origen = fila.index[0]
    st.session_state.anio_final_usuario = int(fila["Año_final"].iloc[0])
    nueva = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(n_ventana)])
    for var in indicadores:
        for i in range(n_ventana):
            nueva.loc[f"Año {i+1}", var] = fila[f"{var}_-{i}"].values[0]
    st.session_state.df_input = nueva

# --- Entrada editable ---
st.subheader("📜 Ingrese los 17 indicadores financieros (5 años)")
df_input = st.data_editor(st.session_state.df_input, use_container_width=True, num_rows="fixed")

# --- Funciones de distancia ---
def distancia_base(f1, f2, lambda_p, n, pesos):
    total, suma_pesos = 0, 0
    for var in indicadores:
        v1 = [f1.get(f"{var}_-{i}", np.nan) for i in range(n)]
        v2 = [f2.get(f"{var}_-{i}", np.nan) for i in range(n)]
        l1, validos = 0, 0
        for a, b in zip(v1, v2):
            if pd.notna(a) and pd.notna(b):
                if np.isinf(a) and np.isinf(b) and a == b:
                    l1 += 0
                elif np.isinf(a) or np.isinf(b):
                    l1 += np.inf
                else:
                    l1 += abs(a - b)
                validos += 1
        penalizada = l1 * (1 + lambda_p * ((n - validos) / n)) if validos > 0 else 1.0
        acotada = penalizada / (1 + penalizada) if np.isfinite(penalizada) else 1.0
        total += pesos[var] * acotada
        suma_pesos += pesos[var]
    return total / suma_pesos if suma_pesos > 0 else 1.0

def distancia_ext(f1, f2, lambda_p, n, pesos):
    base = distancia_base(f1, f2, lambda_p, n, pesos)
    d_dep = 0 if f1["DEP"] == f2["DEP"] else 1
    d_ciiu = 0 if f1["CIIU_Letra"] == f2["CIIU_Letra"] else 1
    desfase = abs(f1["Año_final"] - f2["Año_final"])
    total = base * sum(pesos[var] for var in indicadores)
    total += pesos.get("DEP", 1.0) * d_dep
    total += pesos.get("CIIU", 1.0) * d_ciiu
    total += pesos.get("desfase", 1.0) * desfase
    suma_pesos = sum(pesos[var] for var in indicadores) + pesos.get("DEP",1)+pesos.get("CIIU",1)+pesos.get("desfase",1)
    return total / suma_pesos if suma_pesos > 0 else 1.0

# --- Predicción ---
if st.button("🔍 Predecir riesgo de quiebra"):
    if df_input.isnull().all().all():
        st.warning("⚠️ Debe ingresar al menos un dato numérico.")
    else:
        trayectoria = {f"{var}_-{i}": df_input.loc[f"Año {i+1}", var] for var in indicadores for i in range(n_ventana)}
        trayectoria_ext = trayectoria.copy()
        trayectoria_ext["DEP"] = dep_usuario
        trayectoria_ext["CIIU_Letra"] = ciiu_usuario
        trayectoria_ext["Año_final"] = st.session_state.anio_final_usuario

        # --- Modelo base ---
        distB = espacioB.apply(lambda fila: distancia_base(trayectoria, fila, lambda_b, n_ventana, pesos_b), axis=1)
        if st.session_state.nit_origen: distB = distB.drop(st.session_state.nit_origen, errors="ignore")
        vecinos_b = distB.nsmallest(k_b).index
        prob_b = espacioB.loc[vecinos_b, "RQ_final"].mean()

        # --- Modelo extendido ---
        distE = espacioE.apply(lambda fila: distancia_ext(trayectoria_ext, fila, lambda_e, n_ventana, pesos_e), axis=1)
        if st.session_state.nit_origen: distE = distE.drop(st.session_state.nit_origen, errors="ignore")
        vecinos_e = distE.nsmallest(k_e).index
        prob_e = espacioE.loc[vecinos_e, "RQ_final"].mean()

        # --- Mostrar resultados ---
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"Modelo base: **{prob_b:.2%}** de riesgo")
            res_b = espacioB.loc[vecinos_b, ["DEP", "CIIU_Letra", "Año_final"]].copy()
            res_b["NIT"] = vecinos_b
            res_b["Distancia"] = distB.loc[vecinos_b].values
            st.dataframe(res_b.reset_index(drop=True))

        with col2:
            st.info(f"Modelo extendido: **{prob_e:.2%}** de riesgo")
            res_e = espacioE.loc[vecinos_e, ["DEP", "CIIU_Letra", "Año_final"]].copy()
            res_e["NIT"] = vecinos_e
            res_e["Distancia"] = distE.loc[vecinos_e].values
            st.dataframe(res_e.reset_index(drop=True))

        st.session_state.nit_origen = None

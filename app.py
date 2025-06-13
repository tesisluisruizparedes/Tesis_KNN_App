# --- Streamlit app: Solo modelo funcional extendido con sector económico ---
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Configuración general ---
st.set_page_config(page_title="Predicción de Quiebra Extendida", layout="wide")
st.title("📉 Predicción de Quiebra con Modelo Funcional k-NN Extendido")

st.markdown("""
Ingrese los valores de los 17 indicadores financieros para los **últimos 5 años**. Puede dejar valores en blanco (NaN). Seleccione también el Departamento y el **sector económico**. El modelo buscará las trayectorias más similares considerando tanto el comportamiento financiero como el entorno de operación (región, sector y año).
""")

# --- Mapeo sectores ---
mapeo_sectores = {
    'A': 'Agro', 'B': 'Minería', 'C': 'Industria', 'D': 'Electricidad/Gas',
    'E': 'Agua y desechos', 'F': 'Construcción', 'G': 'Comercio/Vehículos',
    'H': 'Transporte', 'I': 'Turismo/Comida', 'J': 'Comunicaciones',
    'K': 'Finanzas', 'L': 'Inmobiliario', 'M': 'Profesionales',
    'N': 'Servicios adm.', 'O': 'Gobierno', 'P': 'Educación', 'Q': 'Salud',
    'R': 'Entretenimiento', 'S': 'Otros servicios', 'T': 'Hogares',
    'U': 'Entidades ext.', np.nan: 'Sin clasificar'
}
sector_to_letra = {v: k for k, v in mapeo_sectores.items() if pd.notna(k)}

# --- Cargar base extendida y parámetros ---
@st.cache_data
def cargar_datos():
    base_e = pd.read_parquet("20_1EspacioF.parquet")
    with open("20_2ParametrosFuncional.pkl", "rb") as f:
        params_e = pickle.load(f)
    return base_e, params_e

espacioE, paramsE = cargar_datos()

k = paramsE['k']
lambda_p = paramsE['lambda']
pesos = paramsE['pesos']
columnas_funcionales = [col for col in espacioE.columns if "_-" in col]
indicadores = sorted(set(col.split("_-")[0] for col in columnas_funcionales))
n_ventana = len(set(col.split("_-")[1] for col in columnas_funcionales))

# --- Inicializar entrada y variables categóricas ---
if "df_input" not in st.session_state:
    st.session_state.df_input = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(n_ventana)])
if "nit_origen" not in st.session_state:
    st.session_state.nit_origen = None
if "anio_final_usuario" not in st.session_state:
    st.session_state.anio_final_usuario = 2023
if "dep_usuario" not in st.session_state:
    st.session_state.dep_usuario = espacioE["DEP"].dropna().unique()[0]
if "ciiu_usuario" not in st.session_state:
    st.session_state.ciiu_usuario = 'I'  # default: Turismo

# --- Selectores para variables categóricas ---
st.sidebar.subheader("📌 Variables cualitativas")
st.session_state.dep_usuario = st.sidebar.selectbox("Departamento", sorted(espacioE["DEP"].dropna().unique()), index=0, key="dep")
sector_visible = st.sidebar.selectbox("Sector económico", sorted(sector_to_letra.keys()), key="sector")
st.session_state.ciiu_usuario = sector_to_letra[sector_visible]

# --- Botón para cargar trayectoria real ---
if st.sidebar.button("🎯 Usar trayectoria real de ejemplo"):
    fila = espacioE.sample(1)
    st.session_state.nit_origen = fila.index[0]
    st.session_state.anio_final_usuario = int(fila["Año_final"].iloc[0])
    st.session_state.dep_usuario = fila["DEP"].iloc[0]
    st.session_state.ciiu_usuario = fila["CIIU_Letra"].iloc[0]
    nueva = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(n_ventana)])
    for var in indicadores:
        for i in range(n_ventana):
            nueva.loc[f"Año {i+1}", var] = fila[f"{var}_-{i}"].values[0]
    st.session_state.df_input = nueva

# --- Entrada editable ---
st.subheader("📝 Ingrese los 17 indicadores financieros (5 años)")
df_input = st.data_editor(st.session_state.df_input, use_container_width=True, num_rows="fixed")

# --- Función de distancia extendida ---
def distancia_ponderada_extendida(f1, f2, lambda_p, n, pesos):
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
    d_dep = 0 if f1["DEP"] == f2["DEP"] else 1
    d_ciiu = 0 if f1["CIIU_Letra"] == f2["CIIU_Letra"] else 1
    desfase = abs(f1["Año_final"] - f2["Año_final"])
    total += pesos.get("DEP", 1.0) * d_dep
    total += pesos.get("CIIU", 1.0) * d_ciiu
    total += pesos.get("desfase", 1.0) * desfase
    suma_pesos += pesos.get("DEP", 1.0) + pesos.get("CIIU", 1.0) + pesos.get("desfase", 1.0)
    return total / suma_pesos if suma_pesos > 0 else 1.0

# --- Predicción ---
if st.button("🔍 Predecir riesgo de quiebra"):
    if df_input.isnull().all().all():
        st.warning("⚠️ Debe ingresar al menos un dato numérico.")
    else:
        trayectoria = {f"{var}_-{i}": df_input.loc[f"Año {i+1}", var] for var in indicadores for i in range(n_ventana)}
        trayectoria["DEP"] = st.session_state.dep_usuario
        trayectoria["CIIU_Letra"] = st.session_state.ciiu_usuario
        trayectoria["Año_final"] = st.session_state.anio_final_usuario

        distE = espacioE.apply(lambda fila: distancia_ponderada_extendida(trayectoria, fila, lambda_p, n_ventana, pesos), axis=1)
        if st.session_state.nit_origen:
            distE = distE.drop(st.session_state.nit_origen, errors="ignore")
            st.session_state.nit_origen = None

        vecinos_idx = distE.nsmallest(k).index
        prob = espacioE.loc[vecinos_idx, "RQ_final"].mean()

        st.success(f"🔮 Riesgo estimado de quiebra: **{prob:.2%}**")

        resultado = espacioE.loc[vecinos_idx, ["DEP", "CIIU_Letra", "Año_final"]].copy()
        resultado["NIT"] = vecinos_idx
        resultado["Sector económico"] = resultado["CIIU_Letra"].map(mapeo_sectores)
        resultado.drop(columns=["CIIU_Letra", "Año_final"], inplace=True)
        resultado["Distancia funcional"] = distE.loc[vecinos_idx].values
        resultado = resultado[["NIT", "DEP", "Sector económico", "Distancia funcional"]]
        st.dataframe(resultado.reset_index(drop=True), use_container_width=True)


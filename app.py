import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- Configuración ---
st.set_page_config(page_title="Predicción de Quiebra", layout="wide")
st.title("📉 Predicción con Modelo Funcional k-NN (Versión Final)")


# --- Explicación ---
st.markdown("""
Ingrese los valores de los 17 indicadores financieros para los **últimos 5 años**.  
Puede dejar valores en blanco (NaN).  
El modelo calculará la distancia funcional con cada empresa histórica y devolverá:

- La probabilidad estimada de quiebra.
- Los vecinos más cercanos (NIT y año final).
""")

# --- Cargar datos necesarios ---
@st.cache_data
def cargar_base():
    espacioF = pd.read_parquet("9_1EspacioF.parquet")
    with open("9_2ParametrosFuncional.pkl", "rb") as f:
        parametros = pickle.load(f)
    return espacioF, parametros

espacioF, params = cargar_base()
k = params["k"]
lambda_p = params["lambda"]
pesos = params["pesos"]
columnas_funcionales = [col for col in espacioF.columns if "_-" in col]
indicadores = sorted(set(col.split("_")[0] for col in columnas_funcionales))
n_ventana = len(set(col.split("_")[1] for col in columnas_funcionales))

st.sidebar.markdown(f"""
**🔧 Parámetros del modelo**
- k = {k}  
- λ = {lambda_p:.4f}  
- Indicadores: {len(indicadores)}  
- Ventana: {n_ventana} años
""")

# --- Entrada editable ---
st.subheader("📝 Ingrese los 17 indicadores financieros (últimos 5 años)")
df_input = pd.DataFrame(columns=indicadores, index=[f"Año {i+1}" for i in range(n_ventana)])
df_input = st.data_editor(df_input, use_container_width=True, num_rows="fixed")

st.write("Ejemplo de fila de espacioF:")
st.write(espacioF.iloc[0])


# --- Métrica funcional personalizada ---
def distancia_ponderada(f1, f2, lambda_p, n, pesos):
    f1 = f1.to_dict() if isinstance(f1, pd.Series) else f1
    f2 = f2.to_dict() if isinstance(f2, pd.Series) else f2
    total, suma_pesos = 0, 0

    for var in indicadores:
        v1 = [f1.get(f"{var}_-{i}", np.nan) for i in range(n)]
        v2 = [f2.get(f"{var}_-{i}", np.nan) for i in range(n)]
        l1, validos = 0, 0

        for a, b in zip(v1, v2):
            try:
                if pd.notna(a) and pd.notna(b):
                    if np.isinf(a) and np.isinf(b) and a == b:
                        l1 += 0
                    elif np.isinf(a) or np.isinf(b):
                        l1 += np.inf
                    else:
                        l1 += abs(a - b)
                    validos += 1
            except TypeError:
                # Si a o b son None o tipos no numéricos, simplemente los ignoramos
                continue

        faltantes = n - validos
        if validos > 0:
            penalizada = l1 * (1 + lambda_p * (faltantes / n))
            acotada = penalizada / (1 + penalizada) if np.isfinite(penalizada) else 1.0
        else:
            acotada = 1.0

        total += pesos[var] * acotada
        suma_pesos += pesos[var]

    return total / suma_pesos if suma_pesos > 0 else 1.0



# --- Predicción ---
if st.button("🔍 Predecir riesgo de quiebra"):
    if df_input.isnull().all().all():
        st.warning("⚠️ Debe ingresar al menos un dato para calcular distancia.")
    else:
        # Crear diccionario con nombres del tipo VAR_-0, VAR_-1, ..., VAR_-4
        trayectoria = {}
        for i in range(n_ventana):
            for var in indicadores:
                trayectoria[f"{var}_-{i}"] = df_input.loc[f"Año {i+1}", var]

        st.info("⏳ Calculando distancias funcionales...")
        distancias = espacioF.apply(lambda fila: distancia_ponderada(trayectoria, fila, lambda_p, n_ventana, pesos), axis=1)
        vecinos_idx = distancias.nsmallest(k).index
        prob_quiebra = espacioF.loc[vecinos_idx, "RQ_final"].mean()

        st.success(f"🔮 Probabilidad estimada de quiebra: **{prob_quiebra:.2%}**")

        st.markdown("### 🧭 Empresas más similares")
        resultado = espacioF.loc[vecinos_idx, ["DEP", "CIIU_Letra", "Año_final"]].copy()
        resultado["NIT"] = vecinos_idx
        resultado["Distancia funcional"] = distancias.loc[vecinos_idx].values
        resultado = resultado[["NIT", "Año_final", "DEP", "CIIU_Letra", "Distancia funcional"]]
        st.dataframe(resultado.reset_index(drop=True), use_container_width=True)

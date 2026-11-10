# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:43:22 2026

@author: maria
"""

import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de la página y título
st.set_page_config(page_title="IA Explicable - Neumología HUPA", layout="centered")
st.title("🫁 Calculadora de Riesgo en Bronquiectasias")
st.markdown("### Demostración práctica de IA Explicable con XGBoost y SHAP")

# 2. Simulación rápida de un modelo y datos de entrenamiento
@st.cache_resource
def cargar_modelo_y_explainer():
    np.random.seed(42)
    X_train = pd.DataFrame({
        'fev1': np.random.randint(30, 90, 100),
        'exac_previas': np.random.randint(0, 6, 100),
        'pseudomonas': np.random.choice([0, 1], 100),
        'edad': np.random.randint(40, 80, 100)
    })
    y_train = (X_train['exac_previas'] > 2).astype(int) | (X_train['pseudomonas'] == 1).astype(int)
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_train

model, explainer, X_train = cargar_modelo_y_explainer()

# 3. Barra lateral con los controles clínicos (Inputs del paciente)
st.sidebar.header("Parámetros Clínicos del Paciente")
fev1_val = st.sidebar.slider("FEV1 (%)", 20, 100, 52)
exac_val = st.sidebar.number_input("Exacerbaciones año previo", 0, 10, 3)
pseudo_val = st.sidebar.selectbox("Colonización por Pseudomonas", ["No (0)", "Sí (1)"])
pseudo_num = 1 if "Sí" in pseudo_val else 0
edad_val = st.sidebar.number_input("Edad (años)", 18, 90, 68)

paciente_df = pd.DataFrame({
    'fev1': [fev1_val],
    'exac_previas': [exac_val],
    'pseudomonas': [pseudo_num],
    'edad': [edad_val]
})

# 4. Predicción del modelo
probabilidad = model.predict_proba(paciente_df)[0][1] * 100

st.subheader("Resultado de la Predicción")
if probabilidad > 50:
    st.error(f"⚠️ Riesgo Predicho de Mala Evolución / Ingreso: **{probabilidad:.1f}% (Alto Riesgo)**")
else:
    st.success(f"✅ Riesgo Predicho de Mala Evolución / Ingreso: **{probabilidad:.1f}% (Bajo Riesgo)**")

# 5. Cálculo y visualización de Valores SHAP (Explicabilidad Local)
st.markdown("---")
st.subheader("🔍 Explicabilidad Local (Gráfico SHAP - Waterfall)")
st.markdown("Este gráfico muestra cómo cada variable clínica empuja el riesgo del paciente hacia arriba (rojo) o hacia abajo (azul) en comparación con la media de la cohorte.")

shap_values = explainer(paciente_df)

fig, ax = plt.subplots(figsize=(8, 4))
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

st.info("💡 **Mensaje para la sesión clínica:** El modelo transparente nos permite justificar de manera objetiva por qué este paciente concreto requiere una intensificación del tratamiento.")
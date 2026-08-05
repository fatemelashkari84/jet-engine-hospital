import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Jet Engine Hospital", layout="wide")
st.title("Jet Engine Hospital")
st.subheader("Predicting Failure Before It Happens")

@st.cache_resource
def load_artifacts():
    return joblib.load('dashboard_artifacts.pkl')

artifacts = load_artifacts()
data = artifacts['test_data']
feature_cols = artifacts['feature_cols']

def safe_float(value):
    if isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    return float(value)

def dashboard(engine_id, cycle):
    engine_data = data[data['unit_number'] == engine_id].sort_values('time_cycles')
    if len(engine_data) == 0:
        return None, "Engine not found"
    
    row = engine_data[engine_data['time_cycles'] == cycle]
    if len(row) == 0:
        return None, "Cycle not found"
    
    features = row[feature_cols].values[0]
    
    model = artifacts['best_rf_model']
    rul_mean = float(model.predict(features.reshape(1, -1))[0])
    rul_lower = safe_float(artifacts['rul_interval_lower'])
    rul_upper = safe_float(artifacts['rul_interval_upper'])
    
    lr_model = artifacts['lr_model']
    calibrators = artifacts['calibrators']
    proba = float(lr_model.predict_proba(features.reshape(1, -1))[0][1])
    proba_10 = float(calibrators['iso_10'].predict([proba])[0])
    proba_20 = float(calibrators['iso_20'].predict([proba])[0])
    proba_30 = float(calibrators['iso_30'].predict([proba])[0])
    
    scaler = artifacts['scaler']
    iso_model = artifacts['iso_model']
    features_scaled = scaler.transform(features.reshape(1, -1))
    iso_score = float(-iso_model.score_samples(features_scaled)[0])
    
    warnings = []
    confidence = 'High'
    if rul_lower < 10:
        warnings.append(f'Critical RUL: {rul_lower:.0f} cycles')
    if proba_20 > 0.7:
        warnings.append(f'High failure risk: {proba_20:.2f}')
    if iso_score > safe_float(artifacts['iso_threshold']):
        warnings.append(f'Anomaly detected: {iso_score:.3f}')
    if rul_upper - rul_lower > 50:
        confidence = 'Medium'
    
    if len(warnings) == 0:
        action, explanation = 'CONTINUE', 'No critical signals detected.'
    elif rul_lower < 10 or len(warnings) >= 2:
        action, explanation = 'STOP', f'Critical signals: {", ".join(warnings)}'
    else:
        action, explanation = 'INSPECT', f'Elevated risk: {", ".join(warnings)}'
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(engine_data['time_cycles'], engine_data['sensor_2'], label='sensor_2', color='blue')
    ax.axvline(x=cycle, color='red', linestyle='--', label=f'Selected Cycle: {cycle}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Sensor Value')
    ax.set_title(f'Engine {engine_id} - Sensor 2 Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, action, explanation, confidence, rul_mean, rul_lower, rul_upper, proba_10, proba_20, proba_30, iso_score

st.sidebar.header("Input Parameters")
engine_id = st.sidebar.number_input("Engine ID", min_value=1, max_value=100, value=1, step=1)
cycle = st.sidebar.number_input("Cycle", min_value=1, value=1, step=1)

if st.sidebar.button("Analyze"):
    with st.spinner("Analyzing engine data..."):
        result = dashboard(engine_id, cycle)
        if result[1] == "Engine not found" or result[1] == "Cycle not found":
            st.error(result[1])
        else:
            fig, action, explanation, confidence, rul_mean, rul_lower, rul_upper, proba_10, proba_20, proba_30, iso_score = result
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RUL (cycles)", f"{rul_mean:.0f}")
                st.metric("90% Prediction Interval", f"{rul_lower:.0f} - {rul_upper:.0f}")
            with col2:
                st.metric("Failure Risk (10)", f"{proba_10:.3f}")
                st.metric("Failure Risk (20)", f"{proba_20:.3f}")
                st.metric("Failure Risk (30)", f"{proba_30:.3f}")
            with col3:
                st.metric("Anomaly Score", f"{iso_score:.3f}")
            
            st.subheader("Recommendation")
            if action == "CONTINUE":
                st.success(f"Action: {action}")
            elif action == "INSPECT":
                st.warning(f"Action: {action}")
            else:
                st.error(f"Action: {action}")
            st.write(f"Explanation: {explanation}")
            st.write(f"Confidence: {confidence}")
            
            st.pyplot(fig)


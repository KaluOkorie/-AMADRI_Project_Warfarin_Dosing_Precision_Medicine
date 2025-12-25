# streamlit_app_simple.py - NO SHAP DEPENDENCY
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# PAGE CONFIG (MUST BE FIRST)
# ============================================================
st.set_page_config(
    page_title="Warfarin Dosing Assistant",
    page_icon="💊",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================
DOSE_CATEGORIES = [
    (0, 2, "🔴", "VERY LOW"),
    (2, 3, "🟡", "LOW"),
    (3, 7, "🟢", "STANDARD"),
    (7, 10, "🟠", "HIGH"),
    (10, float("inf"), "🔴", "VERY HIGH"),
]

# ============================================================
# SIMPLE MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_simple():
    """Load model without SHAP"""
    try:
        BASE_DIR = Path(__file__).parent
        model_path = BASE_DIR / "final_warfarin_model_xgboost.pkl"
        preprocessor_path = BASE_DIR / "preprocessor.pkl"
        
        if not model_path.exists():
            st.error("❌ Model file not found")
            return None, None
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

# ============================================================
# GENETIC MAPPINGS
# ============================================================
CYP2C9_UI_TO_RAW = {
    "Very high metabolism": "*1/*1",
    "High metabolism": "*1/*2", 
    "Moderate metabolism": "*1/*3",
    "Moderate-low metabolism": "*2/*2",
    "Low metabolism": "*2/*3",
    "Very low metabolism": "*3/*3",
}

VKORC1_UI_TO_RAW = {
    "Standard sensitivity": "G/G",
    "Intermediate sensitivity": "A/G",
    "High sensitivity": "A/A",
}

CYP4F2_UI_TO_RAW = {
    "No variant": "C/C",
    "One variant copy": "C/T",
    "Two variant copies": "T/T",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_dose(dose):
    for low, high, icon, label in DOSE_CATEGORIES:
        if low <= dose < high:
            return icon, label
    return "❓", "UNKNOWN"

def calc_bmi(weight, height_cm):
    h = height_cm / 100
    return weight / (h * h)

def calc_bsa(weight, height_cm):
    return 0.007184 * (height_cm ** 0.725) * (weight ** 0.425)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("💊 Warfarin Dosing Calculator")
    st.markdown("*Educational tool for pharmacogenetic dosing*")
    
    # Load model
    model, preprocessor = load_model_simple()
    if model is None:
        st.warning("Running in demo mode without model")
    
    # Input form
    with st.form("inputs"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.slider("Age (years)", 18, 100, 65)
            weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, step=0.5)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=1.0)
            sex = st.radio("Sex", ["M", "F"], horizontal=True)
            ethnicity = st.selectbox("Ethnicity", 
                ["Caucasian", "Asian", "African American", "Hispanic", "Other"])
            egfr = st.slider("eGFR (ml/min/1.73m²)", 5.0, 200.0, 90.0, step=1.0)
        
        with col2:
            st.subheader("Genetic Profile")
            cyp2c9 = st.selectbox("CYP2C9", list(CYP2C9_UI_TO_RAW.keys()))
            vkorc1 = st.selectbox("VKORC1", list(VKORC1_UI_TO_RAW.keys()))
            cyp4f2 = st.selectbox("CYP4F2", list(CYP4F2_UI_TO_RAW.keys()))
            
            st.subheader("Medications")
            col2a, col2b = st.columns(2)
            with col2a:
                amio = st.checkbox("Amiodarone")
                abx = st.checkbox("Antibiotics")
            with col2b:
                statin = st.checkbox("Statin")
                aspirin = st.checkbox("Aspirin")
            
            st.subheader("Conditions")
            col2c, col2d = st.columns(2)
            with col2c:
                hypertension = st.checkbox("Hypertension")
                diabetes = st.checkbox("Diabetes")
            with col2d:
                ckd = st.checkbox("CKD")
                hf = st.checkbox("Heart Failure")
        
        submitted = st.form_submit_button("Calculate Dose", type="primary")
    
    # When submitted
    if submitted and model is not None:
        with st.spinner("Calculating..."):
            # Calculate features
            bmi = calc_bmi(weight, height)
            bsa = calc_bsa(weight, height)
            
            # Map genotypes
            cyp2c9_raw = CYP2C9_UI_TO_RAW[cyp2c9]
            vkorc1_raw = VKORC1_UI_TO_RAW[vkorc1]
            cyp4f2_raw = CYP4F2_UI_TO_RAW[cyp4f2]
            
            # Create input row (simplified)
            row = {
                "Age": age,
                "Sex": sex,
                "Weight_kg": weight,
                "Height_cm": height,
                "Ethnicity": ethnicity,
                "Hypertension": int(hypertension),
                "Diabetes": int(diabetes),
                "Chronic_Kidney_Disease": int(ckd),
                "Heart_Failure": int(hf),
                "Amiodarone": int(amio),
                "Antibiotics": int(abx),
                "Aspirin": int(aspirin),
                "Statins": int(statin),
                "CYP2C9": cyp2c9_raw,
                "VKORC1": vkorc1_raw,
                "CYP4F2": cyp4f2_raw,
                "BMI": bmi,
                "BSA": bsa,
                "eGFR": egfr,
            }
            
            # Add required columns
            for col in preprocessor.feature_names_in_:
                if col not in row:
                    row[col] = 0 if "num__" in str(col) else "Unknown"
            
            # Create DataFrame
            df = pd.DataFrame([row])
            X = preprocessor.transform(df)
            
            # Predict
            pred = float(model.predict(X)[0])
            icon, category = classify_dose(pred)
            
            # Display results
            st.divider()
            
            # Dose display
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; 
                         background-color: #f8f9fa; border-radius: 10px;'>
                    <h1 style='margin: 0;'>{icon} {pred:.2f} mg/day {icon}</h1>
                    <h3 style='margin: 5px 0; color: #666;'>{category}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Patient summary
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Summary")
                st.write(f"**Age:** {age} years")
                st.write(f"**Sex:** {sex}")
                st.write(f"**Ethnicity:** {ethnicity}")
                st.write(f"**BMI:** {bmi:.1f} kg/m²")
                st.write(f"**BSA:** {bsa:.2f} m²")
                st.write(f"**eGFR:** {egfr:.0f}")
            
            with col2:
                st.subheader("Genetic Profile")
                st.write(f"**CYP2C9:** {cyp2c9}")
                st.write(f"**VKORC1:** {vkorc1}")
                st.write(f"**CYP4F2:** {cyp4f2}")
                
                st.subheader("Clinical Factors")
                st.write(f"**Medications:** {amio+abx+statin+aspirin} interacting")
                st.write(f"**Conditions:** {hypertension+diabetes+ckd+hf} present")
            
            # Disclaimer
            st.divider()
            st.warning("""
            ⚠️ **Educational Use Only**
            This tool provides estimates based on population data.
            Not for clinical decision-making. Always consult healthcare professionals.
            """)
    
    elif submitted:
        # Demo mode
        st.info("Running in demo mode. Install model files for predictions.")
        st.code("""
        Required files in same directory:
        - final_warfarin_model_xgboost.pkl
        - preprocessor.pkl
        """)

if __name__ == "__main__":
    main()

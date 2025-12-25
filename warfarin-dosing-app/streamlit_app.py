# streamlit_app.py - Warfarin Dosing with SHAP Fallback
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Set page config (MUST BE FIRST)
st.set_page_config(
    page_title="Pharmacogenetic Warfarin Dosing",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SHAP IMPORT WITH GRACEFUL FALLBACK
# ============================================================

SHAP_AVAILABLE = False
explainer = None

try:
    import shap
    SHAP_AVAILABLE = True
    st.success("✅ SHAP loaded successfully")
except ImportError as e:
    st.warning("⚠️ SHAP not available. Feature explanations disabled.")
    st.info("Install with: `pip install shap`")

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

MODEL_METADATA = {
    "version": "2.1.0",
    "training_date": "2024-01-20",
    "model_type": "XGBoost",
    "validation_RMSE": 1.15,
    "validation_MAE": 0.85,
    "validation_R2": 0.75
}

# ============================================================
# CACHED MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """Load and cache models"""
    try:
        BASE_DIR = Path(__file__).parent
        
        model_path = BASE_DIR / "final_warfarin_model_xgboost.pkl"
        preprocessor_path = BASE_DIR / "preprocessor.pkl"
        
        # Check if files exist
        if not model_path.exists():
            # Try to find with different patterns
            pkl_files = list(BASE_DIR.glob("*.pkl"))
            if pkl_files:
                st.info(f"Found PKL files: {[f.name for f in pkl_files]}")
            raise FileNotFoundError(f"Model file not found. Looking for: {model_path}")
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        # Load models
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = None
        
        RAW_COLUMNS = list(preprocessor.feature_names_in_)
        
        return model, preprocessor, explainer, RAW_COLUMNS
        
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()

# Load models
try:
    model, preprocessor, explainer, RAW_COLUMNS = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {str(e)}")
    st.info("Please ensure model files are in the same directory as this script.")
    st.code("""
    Required files:
    - final_warfarin_model_xgboost.pkl
    - preprocessor.pkl
    """)
    st.stop()

# ============================================================
# GENETIC MAPPINGS
# ============================================================

CYP2C9_ACTIVITY = {
    "*1/*1": 2.0, "*1/*2": 1.5, "*1/*3": 1.0,
    "*2/*2": 1.0, "*2/*3": 0.5, "*3/*3": 0.0,
}

CYP2C9_PHENO = {
    "*1/*1": "Extensive",
    "*1/*2": "Intermediate",
    "*1/*3": "Intermediate",
    "*2/*2": "Poor",
    "*2/*3": "Poor",
    "*3/*3": "Poor",
}

VKORC1_SENS = {"G/G": 0.0, "A/G": 1.0, "A/A": 2.0}
VKORC1_PHENO = {"G/G": "Normal", "A/G": "Intermediate", "A/A": "Sensitive"}

CYP4F2_SCORE = {"C/C": 0.0, "C/T": 1.0, "T/T": 2.0}
CYP4F2_PHENO = {"C/C": "Normal", "C/T": "Heterozygous", "T/T": "Variant"}

# User-friendly mappings
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
    """Classify dose into clinical categories"""
    for low, high, icon, label in DOSE_CATEGORIES:
        if low <= dose < high:
            return icon, label
    return "❓", "UNKNOWN"

def calc_bsa(weight, height_cm):
    """Calculate Body Surface Area (DuBois formula)"""
    return 0.007184 * (height_cm ** 0.725) * (weight ** 0.425)

def calc_bmi(weight, height_cm):
    """Calculate Body Mass Index"""
    h = height_cm / 100
    return weight / (h * h)

def bmi_category(bmi):
    """Classify BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def renal_impairment(eGFR):
    """Classify renal impairment"""
    if eGFR < 30:
        return "Severe"
    elif eGFR < 60:
        return "Moderate"
    elif eGFR < 90:
        return "Mild"
    return "Normal"

def genetic_burden(cyp2c9, vkorc1, cyp4f2):
    """Calculate combined genetic burden score"""
    c2_norm = CYP2C9_ACTIVITY[cyp2c9] / 2
    v1_norm = VKORC1_SENS[vkorc1] / 2
    c4_norm = CYP4F2_SCORE[cyp4f2] / 2
    return (c2_norm + v1_norm + c4_norm) / 3

def validate_inputs(age, weight, height, egfr):
    """Validate clinical input ranges"""
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be 18–100 years")
    if not (30 <= weight <= 250):
        errors.append("Weight must be 30–250 kg")
    if not (100 <= height <= 250):
        errors.append("Height must be 100–250 cm")
    if not (5 <= egfr <= 200):
        errors.append("eGFR must be 5–200 ml/min/1.73m²")
    return errors

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(inputs_dict):
    """Core prediction function"""
    # Map UI labels to raw genotypes
    cyp2c9_raw = CYP2C9_UI_TO_RAW[inputs_dict["cyp2c9"]]
    vkorc1_raw = VKORC1_UI_TO_RAW[inputs_dict["vkorc1"]]
    cyp4f2_raw = CYP4F2_UI_TO_RAW[inputs_dict["cyp4f2"]]
    
    # Calculate derived features
    bmi = calc_bmi(inputs_dict["weight"], inputs_dict["height"])
    bsa = calc_bsa(inputs_dict["weight"], inputs_dict["height"])
    burden = genetic_burden(cyp2c9_raw, vkorc1_raw, cyp4f2_raw)
    
    interaction_score = sum([
        inputs_dict["amio"], inputs_dict["abx"],
        inputs_dict["statin"], inputs_dict["aspirin"]
    ])
    
    comorb_score = sum([
        inputs_dict["hypertension"], inputs_dict["diabetes"],
        inputs_dict["ckd"], inputs_dict["hf"]
    ])
    
    # Build feature row
    row = {
        "Age": inputs_dict["age"],
        "Sex": inputs_dict["sex"],
        "Weight_kg": inputs_dict["weight"],
        "Height_cm": inputs_dict["height"],
        "Ethnicity": inputs_dict["ethnicity"],
        "Hypertension": int(inputs_dict["hypertension"]),
        "Diabetes": int(inputs_dict["diabetes"]),
        "Chronic_Kidney_Disease": int(inputs_dict["ckd"]),
        "Heart_Failure": int(inputs_dict["hf"]),
        "Amiodarone": int(inputs_dict["amio"]),
        "Antibiotics": int(inputs_dict["abx"]),
        "Aspirin": int(inputs_dict["aspirin"]),
        "Statins": int(inputs_dict["statin"]),
        "CYP2C9": cyp2c9_raw,
        "VKORC1": vkorc1_raw,
        "CYP4F2": cyp4f2_raw,
        "Alcohol_Intake": "Unknown",
        "Smoking_Status": "Unknown",
        "Diet_VitK_Intake": "Unknown",
        "CYP2C9_pcyp3": 0,
        "VKORC1_AA": 0,
        "CYP2C9_Phenotype": CYP2C9_PHENO[cyp2c9_raw],
        "VKORC1_Phenotype": VKORC1_PHENO[vkorc1_raw],
        "CYP4F2_Genotype": CYP4F2_PHENO[cyp4f2_raw],
        "BMI_Category": bmi_category(bmi),
        "Renal_Impairment": renal_impairment(inputs_dict["egfr"]),
        "CYP2C9_Activity": CYP2C9_ACTIVITY[cyp2c9_raw],
        "VKORC1_Sensitivity": VKORC1_SENS[vkorc1_raw],
        "CYP4F2_Score": CYP4F2_SCORE[cyp4f2_raw],
        "BSA": bsa,
        "BMI": bmi,
        "eGFR": inputs_dict["egfr"],
        "Interaction_Score": interaction_score,
        "Comorbidity_Score": comorb_score,
        "Genetic_Burden_Score": burden,
    }
    
    # Create DataFrame and predict
    df_raw = pd.DataFrame([row])[RAW_COLUMNS]
    X = preprocessor.transform(df_raw)
    pred = float(model.predict(X)[0])
    
    # Get SHAP values if available
    shap_top3 = []
    if SHAP_AVAILABLE and explainer is not None:
        try:
            shap_vals = explainer.shap_values(X)[0]
            names = preprocessor.get_feature_names_out()
            pairs = list(zip(names, shap_vals))
            top3 = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
            shap_top3 = [(feat, float(val)) for feat, val in top3]
        except Exception as e:
            st.warning(f"SHAP calculation failed: {str(e)}")
    
    return {
        "prediction": pred,
        "icon": classify_dose(pred)[0],
        "category": classify_dose(pred)[1],
        "bmi": bmi,
        "bsa": bsa,
        "burden": burden,
        "shap_top3": shap_top3,
        "interaction_score": interaction_score,
        "comorb_score": comorb_score,
        "shap_available": SHAP_AVAILABLE and explainer is not None
    }

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    # Header
    st.title("💊 Pharmacogenetic Warfarin Dosing Assistant")
    st.markdown("""
    *Estimate warfarin maintenance dose using genetic and clinical factors*  
    **For educational and research purposes only**
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(f"""
        **Version:** {MODEL_METADATA['version']}  
        **Type:** {MODEL_METADATA['model_type']}  
        **Trained:** {MODEL_METADATA['training_date']}  
        
        **Validation Metrics:**  
        • RMSE: {MODEL_METADATA['validation_RMSE']} mg/day  
        • MAE: {MODEL_METADATA['validation_MAE']} mg/day  
        • R²: {MODEL_METADATA['validation_R2']}
        """)
        
        st.divider()
        
        # SHAP status
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP not installed")
            st.code("pip install shap")
        elif explainer is None:
            st.warning("⚠️ SHAP explainer not available")
        else:
            st.success("✅ SHAP explanations enabled")
        
        st.divider()
        
        st.header("🧬 Genetic Key")
        st.markdown("""
        **CYP2C9:** Warfarin metabolism  
        **VKORC1:** Warfarin sensitivity  
        **CYP4F2:** Vitamin K handling
        """)
        
        st.divider()
        
        st.warning("""
        ⚠️ **Clinical Disclaimer**  
        Educational/research use only.  
        Not for clinical decision-making.
        """)
    
    # Input form
    with st.form("warfarin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Demographics")
            age = st.slider("Age (years)", 18, 100, 65)
            weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, step=0.1)
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.1)
            sex = st.radio("Sex", ["M", "F"], horizontal=True)
            ethnicity = st.selectbox("Ethnicity", 
                ["African American", "Asian", "Caucasian", "Hispanic", "Other"])
            egfr = st.number_input("eGFR (ml/min/1.73m²)", 5.0, 200.0, 90.0, step=0.1)
        
        with col2:
            st.subheader("🧬 Genetic Profile")
            cyp2c9 = st.selectbox("CYP2C9 (Metabolism)", list(CYP2C9_UI_TO_RAW.keys()))
            vkorc1 = st.selectbox("VKORC1 (Sensitivity)", list(VKORC1_UI_TO_RAW.keys()))
            cyp4f2 = st.selectbox("CYP4F2 (Vitamin K)", list(CYP4F2_UI_TO_RAW.keys()))
            
            st.subheader("💊 Medications")
            col2a, col2b = st.columns(2)
            with col2a:
                amio = st.checkbox("Amiodarone")
                abx = st.checkbox("Recent Antibiotics")
            with col2b:
                statin = st.checkbox("Statin Therapy")
                aspirin = st.checkbox("Aspirin Use")
            
            st.subheader("🏥 Comorbidities")
            col2c, col2d = st.columns(2)
            with col2c:
                hypertension = st.checkbox("Hypertension")
                diabetes = st.checkbox("Diabetes")
            with col2d:
                ckd = st.checkbox("Chronic Kidney Disease")
                hf = st.checkbox("Heart Failure")
        
        # Submit button
        submit_button = st.form_submit_button("🔬 Calculate Warfarin Dose", 
                                            type="primary", use_container_width=True)
    
    # Process prediction
    if submit_button:
        # Validate inputs
        errors = validate_inputs(age, weight, height, egfr)
        if errors:
            st.error("**Validation Errors:**\n\n" + "\n".join(f"• {e}" for e in errors))
            return
        
        # Show loading
        with st.spinner("🤖 Calculating dose estimate..."):
            # Collect inputs
            inputs = {
                "age": int(age),
                "weight": float(weight),
                "height": float(height),
                "cyp2c9": cyp2c9,
                "vkorc1": vkorc1,
                "cyp4f2": cyp4f2,
                "ethnicity": ethnicity,
                "sex": sex,
                "egfr": float(egfr),
                "amio": amio,
                "abx": abx,
                "statin": statin,
                "aspirin": aspirin,
                "hypertension": hypertension,
                "diabetes": diabetes,
                "ckd": ckd,
                "hf": hf
            }
            
            # Make prediction
            result = make_prediction(inputs)
            
            # Display results
            st.divider()
            
            # Dose header
            dose_icon = result["icon"]
            dose_category = result["category"]
            dose_value = result["prediction"]
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f2f6; 
                         border-radius: 10px; border: 2px solid #ddd;'>
                    <h1 style='margin: 0;'>{dose_icon} {dose_value:.2f} mg/day {dose_icon}</h1>
                    <h3 style='margin: 5px 0 0 0; color: #666;'>{dose_category}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Results columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Model Insights")
                
                # SHAP or feature importance
                if result["shap_available"] and result["shap_top3"]:
                    st.markdown("**Top Influencing Factors (SHAP):**")
                    shap_data = []
                    for feat, val in result["shap_top3"]:
                        direction = "Increases" if val > 0 else "Decreases"
                        shap_data.append({
                            "Factor": feat.replace("num__", "").replace("cat__", ""),
                            "Impact": f"{val:.3f}",
                            "Effect": direction
                        })
                    st.table(pd.DataFrame(shap_data))
                else:
                    st.info("ℹ️ Feature explanations require SHAP library")
                    st.code("Install: pip install shap")
                    
                    # Show calculated scores instead
                    st.markdown("**Calculated Risk Scores:**")
                    scores_col1, scores_col2, scores_col3 = st.columns(3)
                    with scores_col1:
                        st.metric("Genetic Burden", f"{result['burden']:.3f}")
                    with scores_col2:
                        st.metric("Interaction Score", result["interaction_score"])
                    with scores_col3:
                        st.metric("Comorbidity Score", result["comorb_score"])
                
                # Warnings
                warnings_list = []
                if egfr < 15:
                    warnings_list.append("⚠️ Severe kidney impairment (eGFR < 15)")
                if age > 80 and dose_value > 5:
                    warnings_list.append("⚠️ Older patient with high estimated dose")
                if result["interaction_score"] >= 3:
                    warnings_list.append("⚠️ Multiple interacting medications")
                
                if warnings_list:
                    with st.expander("Clinical Considerations"):
                        for warning in warnings_list:
                            st.write(warning)
            
            with col2:
                st.subheader("👤 Patient Summary")
                
                # Demographics
                st.markdown("**Demographics**")
                demo_col1, demo_col2 = st.columns(2)
                with demo_col1:
                    st.write(f"• **Age:** {age} years")
                    st.write(f"• **Sex:** {sex}")
                    st.write(f"• **Ethnicity:** {ethnicity}")
                with demo_col2:
                    st.write(f"• **BMI:** {result['bmi']:.1f} kg/m²")
                    st.write(f"• **BSA:** {result['bsa']:.2f} m²")
                    st.write(f"• **eGFR:** {egfr:.1f}")
                
                # Genetics
                st.markdown("**Genetics**")
                st.write(f"• **CYP2C9:** {cyp2c9}")
                st.write(f"• **VKORC1:** {vkorc1}")
                st.write(f"• **CYP4F2:** {cyp4f2}")
                
                # Conditions & Medications
                st.markdown("**Conditions**")
                cond_col1, cond_col2 = st.columns(2)
                with cond_col1:
                    st.write(f"• **HTN:** {'✅' if hypertension else '❌'}")
                    st.write(f"• **Diabetes:** {'✅' if diabetes else '❌'}")
                with cond_col2:
                    st.write(f"• **CKD:** {'✅' if ckd else '❌'}")
                    st.write(f"• **HF:** {'✅' if hf else '❌'}")
                
                st.markdown("**Medications**")
                med_col1, med_col2 = st.columns(2)
                with med_col1:
                    st.write(f"• **Amiodarone:** {'✅' if amio else '❌'}")
                    st.write(f"• **Antibiotics:** {'✅' if abx else '❌'}")
                with med_col2:
                    st.write(f"• **Statins:** {'✅' if statin else '❌'}")
                    st.write(f"• **Aspirin:** {'✅' if aspirin else '❌'}")
            
            st.divider()
            
            # Educational notes
            with st.expander("📚 Educational Information", expanded=False):
                st.markdown("""
                **About Warfarin Dosing:**
                - Target INR range: 2.0-3.0 (2.5-3.5 for mechanical valves)
                - Monitor INR weekly during initiation, then every 4 weeks when stable
                - Adjust dose based on INR trends, not single values
                
                **Genetic Factors:**
                - CYP2C9: Affects warfarin metabolism rate
                - VKORC1: Affects warfarin sensitivity
                - CYP4F2: Affects vitamin K availability
                
                **Important:**
                - This is an educational tool only
                - Individual responses may vary
                - Always follow clinical guidelines
                """)
    
    # Footer
    st.divider()
    st.caption("""
    **Disclaimer:** For educational/research purposes only. Not for clinical use. 
    Model performance: RMSE = 1.15 mg/day, MAE = 0.85 mg/day.
    """)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()

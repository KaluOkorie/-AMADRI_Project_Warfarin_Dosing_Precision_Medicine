# app.py - Streamlit Warfarin Dosing App
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from datetime import datetime

# Set page config (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Pharmacogenetic Warfarin Dosing",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS AND CONFIGURATION
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
    "validation_R2": 0.75,
    "last_loaded": datetime.now().strftime("%Y-%m-%d %H:%M")
}

# ============================================================
# CACHED MODEL LOADING (PERFORMANCE OPTIMIZATION)
# ============================================================

@st.cache_resource
def load_models():
    """Load and cache models for performance"""
    try:
        BASE_DIR = Path(__file__).parent
    except NameError:
        BASE_DIR = Path(os.getcwd())
    
    model_path = BASE_DIR / "final_warfarin_model_xgboost.pkl"
    preprocessor_path = BASE_DIR / "preprocessor.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    if not preprocessor_path.exists():
        st.error(f"❌ Preprocessor file not found at: {preprocessor_path}")
        st.stop()
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    explainer = shap.TreeExplainer(model)
    
    return model, preprocessor, explainer

# Load models
try:
    model, preprocessor, explainer = load_models()
    RAW_COLUMNS = list(preprocessor.feature_names_in_)
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
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
    
    # Get SHAP values
    shap_vals = explainer.shap_values(X)[0]
    names = preprocessor.get_feature_names_out()
    pairs = list(zip(names, shap_vals))
    top3 = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
    
    return {
        "prediction": pred,
        "icon": classify_dose(pred)[0],
        "category": classify_dose(pred)[1],
        "bmi": bmi,
        "bsa": bsa,
        "burden": burden,
        "shap_top3": top3,
        "interaction_score": interaction_score,
        "comorb_score": comorb_score
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
    
    # Sidebar for model info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown(f"""
        **Version:** {MODEL_METADATA['version']}  
        **Type:** {MODEL_METADATA['model_type']}  
        **Trained:** {MODEL_METADATA['training_date']}  
        **Loaded:** {MODEL_METADATA['last_loaded']}  
        
        **Validation Metrics:**  
        • RMSE: {MODEL_METADATA['validation_RMSE']} mg/day  
        • MAE: {MODEL_METADATA['validation_MAE']} mg/day  
        • R²: {MODEL_METADATA['validation_R2']}
        """)
        
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
        This tool is for educational and research purposes only.  
        Not for clinical decision-making.  
        Always consult healthcare professionals.
        """)
    
    # Create input form
    with st.form("warfarin_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Demographics")
            age = st.slider("Age (years)", 18, 100, 65, help="Age range: 18-100 years")
            weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, step=0.1, 
                                   help="Weight range: 30-250 kg")
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, step=0.1,
                                   help="Height range: 100-250 cm")
            sex = st.radio("Sex", ["M", "F"], horizontal=True)
            ethnicity = st.selectbox("Ethnicity", 
                ["African American", "Asian", "Caucasian", "Hispanic", "Other"])
            egfr = st.number_input("eGFR (ml/min/1.73m²)", 5.0, 200.0, 90.0, step=0.1,
                                 help="Estimated Glomerular Filtration Rate")
        
        with col2:
            st.subheader("🧬 Genetic Profile")
            cyp2c9 = st.selectbox("CYP2C9 (Metabolism)", list(CYP2C9_UI_TO_RAW.keys()))
            vkorc1 = st.selectbox("VKORC1 (Sensitivity)", list(VKORC1_UI_TO_RAW.keys()))
            cyp4f2 = st.selectbox("CYP4F2 (Vitamin K)", list(CYP4F2_UI_TO_RAW.keys()))
            
            st.subheader("💊 Medications")
            col2a, col2b = st.columns(2)
            with col2a:
                amio = st.checkbox("Amiodarone", help="Strong CYP2C9 inhibitor")
                abx = st.checkbox("Recent Antibiotics", help="May affect gut flora")
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
    
    # Process when submitted
    if submit_button:
        # Validate inputs
        errors = validate_inputs(age, weight, height, egfr)
        if errors:
            st.error("**Validation Errors:**\n\n" + "\n".join(f"• {e}" for e in errors))
            return
        
        # Show loading spinner
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
            
            # Dose header with color
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
            
            # Results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Model Insights")
                
                # SHAP table
                st.markdown("**Top Influencing Factors:**")
                shap_data = []
                for feat, val in result["shap_top3"]:
                    direction = "Increases" if val > 0 else "Decreases"
                    shap_data.append({
                        "Factor": feat.replace("num__", "").replace("cat__", ""),
                        "Impact": f"{val:.3f}",
                        "Effect": f"{direction} dose"
                    })
                
                st.table(pd.DataFrame(shap_data))
                
                # Risk scores
                st.subheader("📊 Risk Profile")
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Genetic Burden", f"{result['burden']:.3f}")
                with score_col2:
                    st.metric("Interaction Score", result["interaction_score"])
                with score_col3:
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
                
                # Conditions
                st.markdown("**Conditions**")
                cond_col1, cond_col2 = st.columns(2)
                with cond_col1:
                    st.write(f"• **HTN:** {'✅' if hypertension else '❌'}")
                    st.write(f"• **Diabetes:** {'✅' if diabetes else '❌'}")
                with cond_col2:
                    st.write(f"• **CKD:** {'✅' if ckd else '❌'}")
                    st.write(f"• **HF:** {'✅' if hf else '❌'}")
                
                # Medications
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
                - Warfarin dose is adjusted to maintain INR within target range (typically 2.0-3.0)
                - INR monitoring frequency: Weekly during initiation, then every 4 weeks when stable
                - Dose adjustments should be based on INR trends, not single values
                
                **Genetic Considerations:**
                - CYP2C9 variants affect warfarin metabolism rate
                - VKORC1 variants affect warfarin sensitivity
                - CYP4F2 variants affect vitamin K availability
                
                **Important Notes:**
                - This model provides estimates based on population data
                - Individual responses may vary
                - Always follow clinical guidelines and professional judgment
                """)
            
            # Export option
            st.caption("💾 *Results can be saved using browser print/save function*")
    
    # Footer
    st.divider()
    st.caption("""
    **Disclaimer:** This application is for educational and research purposes only. 
    It is not intended for clinical decision-making. Always consult qualified healthcare 
    professionals for medical advice. Model performance: RMSE = 1.15 mg/day, MAE = 0.85 mg/day.
    """)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

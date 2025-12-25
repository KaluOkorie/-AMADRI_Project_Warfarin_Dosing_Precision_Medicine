import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap

# ============================================================
# DOSE CATEGORIES
# ============================================================

DOSE_CATEGORIES = [
    (0, 2, "🔴", "VERY LOW"),
    (2, 3, "🟡", "LOW"),
    (3, 7, "🟢", "STANDARD"),
    (7, 10, "🟠", "HIGH"),
    (10, float("inf"), "🔴", "VERY HIGH"),
]

def classify_dose(dose):
    for low, high, icon, label in DOSE_CATEGORIES:
        if low <= dose < high:
            return icon, label
    return "❓", "UNKNOWN"

# ============================================================
# LOAD MODEL + PREPROCESSOR (SAFE FOR ALL ENVIRONMENTS)
# ============================================================

try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path(os.getcwd())

model_path = BASE_DIR / "final_warfarin_model_xgboost.pkl"
preprocessor_path = BASE_DIR / "preprocessor.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at: {model_path}")
if not preprocessor_path.exists():
    raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
explainer = shap.TreeExplainer(model)

RAW_COLUMNS = list(preprocessor.feature_names_in_)

# ============================================================
# TRAINING-TIME MAPPINGS (CONFIRMED)
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

# ============================================================
# USER-FRIENDLY LABELS → RAW GENOTYPES
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
# TRAINING-TIME FORMULAS
# ============================================================

def calc_bsa(weight, height_cm):
    return 0.007184 * (height_cm ** 0.725) * (weight ** 0.425)

def calc_bmi(weight, height_cm):
    h = height_cm / 100
    return weight / (h * h)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def renal_impairment(eGFR):
    if eGFR < 30:
        return "Severe"
    elif eGFR < 60:
        return "Moderate"
    elif eGFR < 90:
        return "Mild"
    return "Normal"

def genetic_burden(cyp2c9, vkorc1, cyp4f2):
    c2_norm = CYP2C9_ACTIVITY[cyp2c9] / 2
    v1_norm = VKORC1_SENS[vkorc1] / 2
    c4_norm = CYP4F2_SCORE[cyp4f2] / 2
    return (c2_norm + v1_norm + c4_norm) / 3

# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(age, weight, height, egfr):
    errors = []
    if not (18 <= age <= 100):
        errors.append("Age must be 18–100 years.")
    if not (30 <= weight <= 250):
        errors.append("Weight must be 30–250 kg.")
    if not (100 <= height <= 250):
        errors.append("Height must be 100–250 cm.")
    if not (5 <= egfr <= 200):
        errors.append("Kidney function (eGFR) must be 5–200.")
    return errors

# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(
    age, weight, height_cm,
    cyp2c9_ui, vkorc1_ui, cyp4f2_ui,
    ethnicity, sex,
    egfr,
    amio, abx, statin, aspirin,
    hypertension, diabetes, ckd, hf
):
    age = int(age)
    weight = float(weight)
    height_cm = float(height_cm)
    egfr = float(egfr)

    errors = validate_inputs(age, weight, height_cm, egfr)
    if errors:
        return "❌ **Input validation failed:**\n\n" + "\n".join(f"- {e}" for e in errors)

    # Map UI → raw genotypes
    cyp2c9_raw = CYP2C9_UI_TO_RAW[cyp2c9_ui]
    vkorc1_raw = VKORC1_UI_TO_RAW[vkorc1_ui]
    cyp4f2_raw = CYP4F2_UI_TO_RAW[cyp4f2_ui]

    # Derived features
    bmi = calc_bmi(weight, height_cm)
    bsa = calc_bsa(weight, height_cm)
    bmi_cat = bmi_category(bmi)
    renal_cat = renal_impairment(egfr)
    burden = genetic_burden(cyp2c9_raw, vkorc1_raw, cyp4f2_raw)

    interaction_score = int(amio) + int(abx) + int(statin) + int(aspirin)
    comorb_score = int(hypertension) + int(diabetes) + int(ckd) + int(hf)

    # Build RAW dataframe EXACTLY as training expects
    row = {
        "Age": age,
        "Sex": sex,
        "Weight_kg": weight,
        "Height_cm": height_cm,
        "Ethnicity": ethnicity,
        "Hypertension": int(hypertension),
        "Diabetes": int(diabetes),
        "Chronic_Kidney_Disease": int(ckd),
        "Heart_Failure": int(hf),
        "Amiodarone": int(amio),
        "Antibiotics": int(abx),
        "Aspirin": int(aspirin),
        "Statins": int(statin),

        # Raw genotypes
        "CYP2C9": cyp2c9_raw,
        "VKORC1": vkorc1_raw,
        "CYP4F2": cyp4f2_raw,

        # Required but unused columns
        "Alcohol_Intake": "Unknown",
        "Smoking_Status": "Unknown",
        "Diet_VitK_Intake": "Unknown",
        "CYP2C9_pcyp3": 0,
        "VKORC1_AA": 0,

        # Engineered categorical
        "CYP2C9_Phenotype": CYP2C9_PHENO[cyp2c9_raw],
        "VKORC1_Phenotype": VKORC1_PHENO[vkorc1_raw],
        "CYP4F2_Genotype": CYP4F2_PHENO[cyp4f2_raw],
        "BMI_Category": bmi_cat,
        "Renal_Impairment": renal_cat,

        # Engineered numeric
        "CYP2C9_Activity": CYP2C9_ACTIVITY[cyp2c9_raw],
        "VKORC1_Sensitivity": VKORC1_SENS[vkorc1_raw],
        "CYP4F2_Score": CYP4F2_SCORE[cyp4f2_raw],
        "BSA": bsa,
        "BMI": bmi,
        "eGFR": egfr,
        "Interaction_Score": interaction_score,
        "Comorbidity_Score": comorb_score,
        "Genetic_Burden_Score": burden,
    }

    df_raw = pd.DataFrame([row])[RAW_COLUMNS]

    # Apply preprocessor
    X = preprocessor.transform(df_raw)

    # Predict
    pred = float(model.predict(X)[0])
    icon, cls = classify_dose(pred)

    # SHAP
    shap_vals = explainer.shap_values(X)[0]
    names = preprocessor.get_feature_names_out()
    pairs = list(zip(names, shap_vals))
    top3 = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]

    table = "| Feature | SHAP Value | Effect |\n|---------|------------|--------|\n"
    for feat, val in top3:
        eff = "↑ increases dose" if val > 0 else "↓ decreases dose"
        table += f"| {feat} | {val:.3f} | {eff} |\n"

    # Warnings (educational only)
    warnings_list = []
    if egfr < 15:
        warnings_list.append("Severe kidney impairment (eGFR < 15).")
    if age > 80 and pred > 5:
        warnings_list.append("Older patient with a relatively high estimated dose.")
    if interaction_score >= 3:
        warnings_list.append("Several interacting medications may affect INR stability.")

    warn_text = ""
    if warnings_list:
        warn_text = "\n### Clinical-style warnings (educational)\n" + "\n".join(f"- {w}" for w in warnings_list)

    # Output
    return f"""
{icon} **Warfarin dose estimate (model output)** {icon}

**Estimated maintenance dose:** **{pred:.2f} mg/day**  
**Dose band:** {cls}

---

### How the model weighed features (top 3)
{table}

---

### Patient summary
- Age: {age} years  
- Sex: {sex}  
- Ethnicity: {ethnicity}  
- Weight: {weight:.1f} kg  
- Height: {height_cm:.1f} cm  
- BMI: {bmi:.1f} kg/m²  
- BSA (DuBois): {bsa:.2f} m²  
- Kidney function (eGFR): {egfr:.1f} ml/min/1.73m²  
- Combined genetic burden score: {burden:.3f}

### Genetic information (simplified)
- Warfarin metabolism gene (CYP2C9): **{cyp2c9_ui}**  
- Warfarin sensitivity gene (VKORC1): **{vkorc1_ui}**  
- Vitamin K handling gene (CYP4F2): **{cyp4f2_ui}**

### Health conditions
- Hypertension: {bool(hypertension)}  
- Diabetes: {bool(diabetes)}  
- Chronic kidney disease: {bool(ckd)}  
- Heart failure: {bool(hf)}

### Other medicines that may interact
- Amiodarone: {bool(amio)}  
- Recent antibiotics: {bool(abx)}  
- Statin therapy: {bool(statin)}  
- Aspirin use: {bool(aspirin)}

{warn_text}

---

### INR-related educational notes
- Warfarin doses are adjusted to keep INR within a target range.  
- INR is checked more often during initiation or when medicines/conditions change.  
- This model provides an estimate; decisions should always be based on INR results and clinical context.

---

⚠️ **Educational use only — not for clinical decision-making.**
"""

# ============================================================
# GRADIO UI (PATIENT + CLINICIAN FRIENDLY)
# ============================================================

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(18, 100, value=65, step=1, label="Age (years)"),
        gr.Number(value=70.0, label="Weight (kg)", minimum=30, maximum=250),
        gr.Number(value=170.0, label="Height (cm)", minimum=100, maximum=250),

        gr.Radio(list(CYP2C9_UI_TO_RAW.keys()), label="Warfarin metabolism (CYP2C9)"),
        gr.Radio(list(VKORC1_UI_TO_RAW.keys()), label="Warfarin sensitivity (VKORC1)"),
        gr.Radio(list(CYP4F2_UI_TO_RAW.keys()), label="Vitamin K handling (CYP4F2)"),

        gr.Radio(["African American", "Asian", "Caucasian", "Hispanic", "Other"], label="Ethnic background"),
        gr.Radio(["M", "F"], label="Sex"),

        gr.Number(value=90.0, label="Kidney function (eGFR)", minimum=5, maximum=200),

        gr.Checkbox(label="On amiodarone"),
        gr.Checkbox(label="Recent antibiotics"),
        gr.Checkbox(label="On a statin"),
        gr.Checkbox(label="Taking aspirin"),

        gr.Checkbox(label="Hypertension"),
        gr.Checkbox(label="Diabetes"),
        gr.Checkbox(label="Chronic kidney disease"),
        gr.Checkbox(label="Heart failure"),
    ],
    outputs=gr.Markdown(),
    title="Pharmacogenetic Warfarin Dosing Explorer (Educational)",
    description="A user-friendly model explorer for clinicians and patients. Not for prescribing.",
)

if __name__ == "__main__":
    interface.launch(share=True, server_name="0.0.0.0")

# AMADRI Project: Warfarin Dosing Precision Medicine

## Project Overview
I'm excited to share the AMADRI Project, a comprehensive data science initiative to build a precision dosing engine for Warfarin. 
This repository documents a complete, end-to-end workflow that transforms raw clinical data into actionable insights, tackling one of healthcare's most challenging medication management problems.

## Business Problem
Warfarin dosing presents significant clinical and operational challenges:

- Patient Harm: Small dosing errors cause severe bleeding or clotting events
- Financial Strain: Warfarin-related adverse events cost thousands per incident in extended hospital stays
- Operational Inefficiency: Clinicians spend excessive time on manual trial-and-error dosing

## The Precision Dosing Challenge
Warfarin, a life-saving anticoagulant, has remained notoriously difficult to dose correctly due to its narrow therapeutic window. 
Small errors can lead to serious bleeding or clotting events. 
Traditional dosing often relies on slow, iterative adjustments, creating:
- Patient Safety Risks: Unpredictable responses during the stabilization period
- Clinical Burden: Extensive clinician time spent on manual dose titration
- Healthcare Costs: Avoidable hospitalizations from adverse drug events
This project represents a methodical approach to solving this problem through data, building from a solid foundation of 50,000 patient records across clinical, genomic, lifestyle, and outcomes domains.

## Key Design Decisions
The pipeline begins with robust data acquisition, handling API constraints through intelligent design:
- Patient-Centric API Strategy: Individual endpoint calls ensure data alignment across all domains
- Fault-Tolerant Architecture: Checkpointing every 100 patients with resume capability
- Demo Production-Ready Practices: Rate limiting, comprehensive logging, and validation at each stage

## Systematic 5‑Day Development Framework

The project follows a structured progression from raw data to deployable insights:

| Day   | Focus                        | Key Activities                                        | Outcomes                                             |
|-------|------------------------------|-------------------------------------------------------|------------------------------------------------------|
| Day 1 | Data Acquisition             | API integration, consolidation, validation            | 50,000+ patient records across 4 domains             |
| Day 2 | Exploration & Preparation    | EDA, leakage prevention, train‑val‑test splits        | Cleaned datasets, strategic splits for modeling      |
| Day 3 | Feature Engineering & Modeling | Genetic encoding, clinical features, baseline models | Engineered features, model comparison, IWPC benchmark|
| Day 4 | Advanced Modeling & Tracking | XGBoost tuning, neural networks, MLflow tracking      | Optimized models with full experiment reproducibility|
| Day 5 | Explainability & Deployment  | SHAP analysis, clinical validation, Gradio prototype  | Interpretable model with clinical safety guidance    |


## Comprehensive Dataset Architecture

The pipeline successfully consolidated and prepared a substantial dataset for analysis:

| Dataset   | Records | Features | Key Content                                      | Primary Purpose                          |
|-----------|---------|----------|--------------------------------------------------|------------------------------------------|
| Clinical  | 50,000  | 14       | Demographics, comorbidities, concurrent medications | Baseline patient characterization        |
| Genomics  | 50,000  | 4        | CYP2C9, VKORC1, CYP4F2 genotypes                 | Pharmacogenetic determinants             |
| Lifestyle | 50,000  | 4        | Alcohol, smoking, vitamin K intake               | Environmental & behavioral factors       |
| Outcomes  | 50,000  | 5        | Stable dose, INR stabilization, TTR, adverse events | Target variables & performance metrics   |
| Combined  | 50,000  | 27+      | All features merged with engineered additions    | Complete modeling dataset                |

## Clinical Validation
The core modeling approach focused on creating clinically actionable predictions and its backed by relevant literature:

## Feature Engineering with Clinical Wisdom
- CYP2C9 - The "Drug Metabolizer" Gene [Some people break down warfarin quickly (need more), some slowly (need less)]
Raw genotype → Clinical meaning
"*1/*1" = Fast metabolism (needs higher dose)
"*1/*3" = Slow metabolism (needs lower dose)  
"*3/*3" = Very slow metabolism (needs much lower dose)
- VKORC1 - The "Drug Sensitivity" Gene [Some bodies are more sensitive to warfarin - even small doses can cause bleeding]
Raw genotype → Warfarin sensitivity
"G/G" = Normal sensitivity (standard dose)
"A/G" = More sensitive (lower dose)
"A/A" = Very sensitive (much lower dose)
- CYP4F2 - The "Vitamin K Processor" Gene [This causes warfarin to work better, so they need slightly lower doses]
"T/T" variant = Processes vitamin K slower
- Body Wisdom Encoding [Body Mass Index (BMI) tells us about body composition] BMI = weight (kg) / (height (m))²
Body Surface Area (BSA) - better predictor than weight alone
BSA = √[height(cm) × weight(kg) / 3600]
Warfarin spreads through body water, not fat. BSA gives better estimate of distribution space.
Categories:
Underweight (<18.5): Often need lower doses
Normal (18.5-25): Standard doses  
Overweight (25-30): May need slightly higher
Obese (>30): Often need higher doses
- eGFR - "Kidney Filtering Capacity [Kidneys help clear warfarin. Poor kidney function = drug stays longer = needs less.]
Categories:
Severe impairment (<30): Much lower doses
Moderate (30-60): Lower doses  
Mild (60-90): Slightly lower doses
Normal (>90): Standard doses
- Age Adjustment [Liver function declines with age, so drug processing slows.]
Older patients (>75) → Lower doses automatically
- Medication Interaction [Some medications boost warfarin's effects dramatically. Amiodarone alone can double warfarin levels!]
Interaction Score = Amiodarone? + Antibiotics? + Statins? + Aspirin? each "yes" = +1 to score
- Comorbidity Scoring [Multiple health conditions change how the body handles medications.]
Comorbidity Score = Hypertension? + Diabetes? + Kidney Disease? + Heart Failure? each condition = +1 to score 
- Combined Genetic Burden Score [Some patients have multiple genetic "reasons" to need lower doses. This combines them into one risk indicator.]
CYP2C9_score = (2.0 - activity_score) / 2.0
VKORC1_score = sensitivity_score / 2.0  
CYP4F2_score = variant_score / 2.0
Genetic Burden = (CYP2C9 + VKORC1 + CYP4F2) / 3
Higher burden = More genetic reasons for lower doses
## Rigorous Model Evaluation
Benchmarking against the established **IWPC clinical algorithm** showed strong improvements:

| Metric                                    | Result                                |
|-------------------------------------------|---------------------------------------|
| Improvement over IWPC (RMSE)              | **+34.7%**                            |
| Test RMSE                                 | **0.645 mg** (critical for narrow-therapeutic-index drugs) |
| Predictions within ±1.0 mg of actual dose | **84.2%**                             |

## Explainability for Clinical Trust
- **SHAP analysis** revealed clinically coherent feature importance.  
- **VKORC1 sensitivity** and **CYP2C9 activity** emerged as top predictors.  
- Results align with established pharmacological understanding, reinforcing trust in model outputs.

## Deployment
The model is deployed as a Streamlit application on Hugging Face, containerised with Docker and served via a Flask API. 
It is grounded in decades of clinical warfarin research, translating raw patient data into clinically meaningful features. 
This approach helps the model reason more like an experienced clinician, not just a statistical engine.
👉 **Interact with the app:**  
[Click here](https://huggingface.co/spaces/Kalu0147/Warfarin_Dosing_Precision_Medicine#prediction-results)

## Technical Environment
- **Languages:** Python (Pandas, Scikit‑learn, XGBoost, TensorFlow)  
- **MLOps:** Dockerised Streamlit app on Hugging Face, served via Flask; MLflow for experiment tracking 
- **Explainability & Prep:** SHAP for explainability, Scikit‑learn for preprocessing and modeling  
- **Process:** Reproducible, structured 5-day development workflow 

Let's Connect
I'm passionate about building data science solutions that bridge technical excellence with real-world clinical impact.





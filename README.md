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

The core modeling approach focused on creating clinically actionable predictions:

## Feature Engineering with Clinical Wisdom
- Transformed genetic variants into functional activity scores (CYP2C9, VKORC1).
- Calculated clinically relevant metrics including **BSA**, **eGFR**, and **genetic burden scores**.

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

## Deployment & Clinical Integration
The project culminates in a functional Gradio prototype that demonstrates how this technology could integrate into clinical workflows. 
The application provides:
- Personalized dosing recommendations based on genetic and clinical profiles
- Clinical context and safety guidance for each recommendation
- Interpretable explanations of which factors most influenced the dose
- Drug interaction flags for common concomitant medications

## Technical Environment
- **Languages:** Python (Pandas, Scikit‑learn, XGBoost, TensorFlow)  
- **MLOps:** MLflow for experiment tracking, Gradio for deployment  
- **Key Libraries:** SHAP for explainability, Scikit‑learn for preprocessing and modeling  
- **Methodology:** Structured 5‑day development cycle with emphasis on reproducibility  

Let's Connect
I'm passionate about building data science solutions that bridge technical excellence with real-world clinical impact.





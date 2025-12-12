# AMADRI Project: Warfarin Dosing Precision Medicine

## Project Overview
AMADRI is a healthcare data science initiative to develop an AI-driven dosing recommendation engine for Warfarin, 
one of the world's most commonly prescribed anticoagulants with a narrow therapeutic window. 
This repository contains Step 1: Data Acquisition Pipeline of the complete workflow.

## Business Problem
Warfarin dosing presents significant clinical and operational challenges:

- Patient Harm: Small dosing errors cause severe bleeding or clotting events
- Financial Strain: Warfarin-related adverse events cost thousands per incident in extended hospital stays
- Operational Inefficiency: Clinicians spend excessive time on manual trial-and-error dosing

## Business Objectives
- Clinical: Reduce Warfarin-related adverse events by ≥25% within 6-12 months
- Operational: Automate dosing decision support, reducing clinician workload by 30%
- Strategic: Establish market leadership in precision dosing for anticoagulant therapy

## Key Design Decisions
- Patient-Specific API Strategy: Uses individual endpoints (/endpoint/{patient_id}) 
instead of bulk endpoints to bypass API limitations and ensure data alignment
- Adaptive Sampling: Attempts 2x target records to account for incomplete patient data (38.5% success rate observed)
- Checkpoint System: Saves progress every 100 patients for fault tolerance and resume capability
- Rate Limiting: Implements 50ms delays between requests to respect API constraints

## Dataset Strategy
### For Data Analysts (Exploratory Analysis)
Dataset	Records	Key Features	Purpose
- clinical_clean.csv|	3,845	|14 |features incl. demographics, comorbidities, medications	|Baseline patient characterization
- genomics_clean.csv|	3,845	|4 |features (CYP2C9, VKORC1, CYP4F2 genotypes)	               |Pharmacogenomic analysis
- lifestyle_clean.csv|	3,845|	|4 features (alcohol, smoking, vitamin K intake)	          |Lifestyle factor assessment
- combined_clean.csv| 3,845	|20 |features (all above merged)	                               |Comprehensive EDA and reporting
## Pipeline Workflow
### Step 1: Data Acquisition (Current - COMPLETE)
- Status: 3,845 complete patient records acquired
- API Authentication: OAuth2 token-based access to GenexaHealth API
- Patient ID Retrieval: 50,000 patient identifiers obtained
- Strategic Sampling: 4,000 target with adaptive oversampling
- Data Collection: Parallel endpoint calls with fault tolerance
- Validation: Completeness checks across three data domains
- Dataset Creation: Clean CSVs/JSONs for downstream teams





















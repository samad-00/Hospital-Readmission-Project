# 🏥 Hospital 30-Day Readmission Prediction Platform (PySpark ML)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PySpark-3.5%2B-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Project-INT315%20Project%2023-06B6D4?style=for-the-badge" />
</p>

---

## 📌 Executive Overview

**PulseCare ML** is an interactive, enterprise-grade Clinical Decision Support (CDS) and machine learning dashboard designed to predict **30-day unplanned hospital readmissions** for diabetic inpatients using Apache Spark and Scikit-Learn. Built on 10 years (1999–2008) of clinical encounter data across 130 US hospitals from the **UCI Machine Learning Repository**, this platform tackles severe class imbalance (~11.3% early readmissions) to empower hospital care teams, case managers, and clinicians to prevent hospital penalties under the **CMS Hospital Readmissions Reduction Program (HRRP)**.

The interface features brutalist glassmorphic card layouts, real-time hyperparameter tuning, interactive Plotly visualizations, live patient risk calculators, and automated transition-of-care recommendations.

---

## 🎯 INT315 Project 23: Rubric Compliance & Objectives

This project strictly satisfies all requirements specified in **Project 23: Hospital Readmission Prediction**:

| Academic Task Requirement | Implementation Details | Status |
|:---|:---|:---:|
| **1. Load patient discharge records into Spark DataFrame** | Implemented in `src/pyspark_pipeline.py` using `SparkSession`, reading 100k+ clinical records. | ✅ **Complete** |
| **2. Handle null values in admission & discharge fields** | Mapped coded ID fields (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`), imputed blanks/`?`, and filtered out deceased/expired patients. | ✅ **Complete** |
| **3. Encode categorical data (diagnosis type & department)** | Standard ICD-9 clinical categorization (Circulatory, Respiratory, Digestive, Diabetes, Injury, etc.) and physician specialty grouping into structured departments. | ✅ **Complete** |
| **4. Build Decision Tree classifier for readmission likelihood** | Weighted Decision Tree model configured with class balancing (`positive_weight ≈ 7.78`) and depth tuning to handle class imbalance. | ✅ **Complete** |
| **5. Evaluate model using accuracy and ROC-AUC** | Dual evaluation via PySpark `BinaryClassificationEvaluator` and Scikit-Learn pipelines reporting ROC-AUC, Recall, Precision, and F1. | ✅ **Complete** |
| **6. Visualize readmission trends by department and patient age** | Multi-faceted interactive visualizations by Medical Specialty, Patient Age bracket, Length of Stay, and Glycemic control. | ✅ **Complete** |

---

## 🖥️ In-Depth Guide: Every Part of the Dashboard Explained

### 1. 📊 Quick Summary (Executive Overview)
* **What it is**: The high-level executive briefing room showing the 4 primary health metrics and cohort distributions.
* **What it does**: Displays total encounters (100,114), 30-day readmission rate (11.34%), average hospital stay (4.4 days), and model quality score (ROC-AUC ~0.65).
* **Why it is there**: Provides immediate clinical context. It explains the CMS HRRP hospital penalty framework ($17+ billion national impact) and why healthcare systems care about readmissions.
* **How we are doing it**: Aggregating cleaned hospital encounter records and computing percentages and mean stay lengths.
* **How to adjust it**: Use the sidebar **Population Filters** to narrow down by age deciles, departments, or diagnosis groups to observe how metrics change for specific cohorts.
* **How to make it better**: Connect directly to hospital live EHR database streams (HL7/FHIR feeds) for real-time daily admissions updates.

---

### 2. 🔍 Trend Charts (Exploratory Data Analysis - EDA)
* **What it is**: Population-level visual exploration addressing the required task: *"Visualize readmission trends by department and patient age"*.
* **What it does**: Compares 30-day return rates across medical departments, age groups, length of stay, ICD-9 illness categories, and blood sugar (HbA1c) lab tests.
* **Why it is there**: Reveals clinical patterns (e.g. Cardiology and Emergency departments have higher rebound rates; older patients and extended hospital stays are risk amplifiers).
* **How we are doing it**: Grouping records via Pandas/PySpark aggregations and rendering interactive Plotly bar, line, and scatter graphs with hover tooltips.
* **How to adjust it**: Select different departments or age brackets in the sidebar; all 6 charts dynamically re-filter to the selected subpopulation.
* **How to make it better**: Add geographic spatial mapping across different hospital zip codes and socioeconomic vulnerability indices (SVI).

---

### 3. 🧠 Model Testing Lab (5 AI Models Benchmark & Studio)
* **What it is**: The machine learning laboratory where all 5 algorithms are evaluated, tuned, and compared side-by-side.
* **What it does**:
  1. Displays the **Multi-Model Benchmark Matrix** (Decision Tree, Random Forest, Gradient Boosting, Extra Trees, Logistic Regression).
  2. Overlays the multi-model **ROC Curves** against the 50% random-chance baseline.
  3. Provides an interactive **Sensitivity Cutoff Slider (0.10–0.90)** with a dynamic Confusion Matrix (True Positives, False Positives, False Negatives, True Negatives).
  4. Features a **Live Retraining Engine** where you can adjust Max Depth or Number of Trees and click *"Retrain Model Live Now"*.
* **Why it is there**: Fulfills the core modeling task (*Decision Tree classifier evaluated on Accuracy & ROC-AUC*) while demonstrating why **ROC-AUC & Recall** matter far more than raw accuracy on imbalanced data.
* **How we are doing it**: Pre-trained Scikit-Learn pipelines (`ColumnTransformer` + `StandardScaler` + `OneHotEncoder`) serialized in `models/*.joblib`, with live threshold matrix updates.
* **How to adjust it**: Move the **Cutoff Threshold slider**:
  - Drag to `0.35`: Increases Recall (catches 85%+ of sick patients).
  - Drag to `0.55`: Increases raw Accuracy to **85%–88%**.
* **How to make it better**: Implement automated Bayesian hyperparameter search (Optuna) and deep learning models (TabNet / MLP).

---

### 4. 🩺 Patient Risk Checker (Real-Time CDS Predictor)
* **What it is**: A bedside Clinical Decision Support (CDS) simulation tool for physicians and discharge nurses.
* **What it does**: Takes patient vitals, medications, past hospital stays, and diagnoses, calculates their readmission probability (0–100%), displays an interactive Speedometer Gauge, and generates an automated Discharge Action Checklist.
* **Why it is there**: Bridges theoretical machine learning with actionable patient care. An algorithm is useless if clinicians cannot act on its predictions.
* **How we are doing it**: Feeds intake form values into the active trained pipeline's `predict_proba()` method, stratifies risk into tiers (Low <20%, Moderate 20–40%, High >40%), and triggers rule-based clinical transition protocols.
* **How to adjust it**: Test different patient profiles manually, or click one of the preset buttons (*"High-Risk Elderly Heart Patient"* vs *"Low-Risk Planned Surgery Patient"*).
* **How to make it better**: Integrate with hospital pharmacy databases to auto-populate drug-drug interactions and patient insurance coverage.

---

### 5. 📁 Patient List & Export (Cohort Triage)
* **What it is**: A prioritized discharge management worklist for hospital nurse case managers.
* **What it does**: Displays active inpatient encounters sorted by risk score, filters by department or risk threshold, and allows 1-click **CSV/Excel Export**.
* **Why it is there**: Enables case managers to prioritize follow-up calls and home health resources for the top 15% highest-risk patients.
* **How we are doing it**: Filtering pre-computed test cohort dataframes and generating formatted downloadable CSV blobs.
* **How to adjust it**: Drag the *"Show Only Patients with Risk Above"* slider or pick specific clinical departments from the dropdown.
* **How to make it better**: Add automated SMS/email alert triggers directly to patient primary care physicians.

---

### 6. ⚡ Big Data Code (PySpark Architecture Hub)
* **What it is**: The academic compliance and distributed architecture documentation tab.
* **What it does**: Validates all INT315 rubric requirements, explains the necessity of Spark for enterprise EHR data, and displays the full PySpark ML pipeline code.
* **Why it is there**: Proves that the project is built according to big data distributed computing standards.
* **How we are doing it**: Implemented in `src/pyspark_pipeline.py` using native PySpark SQL expressions, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, and `BinaryClassificationEvaluator`.
* **How to adjust it**: Run `python src/pyspark_pipeline.py` directly in the terminal to execute Spark distributed training on your machine.
* **How to make it better**: Deploy the PySpark pipeline onto an AWS EMR or Databricks cloud cluster with Delta Lake storage.

### 4. 🩺 Real-Time Patient Risk Predictor (Clinical CDS)
- **Interactive Intake Simulation**: Input patient demographics, department, admission type, discharge disposition, lab count, medication count, prior inpatient stays, and HbA1c status.
- **One-Click Presets**:
  - *High-Risk Cardiac Geriatric Patient* (Frequent readmissions, insulin adjustment, heart condition)
  - *Low-Risk Elective Surgical Patient* (Routine elective, 0 prior visits)
  - *Custom Patient Intake*
- **Visual Risk Meter**: Dynamic SVG gauge displaying readmission probability with risk stratification badges:
  - 🟢 **LOW RISK** (< 20%)
  - 🟡 **MODERATE RISK** (20% – 40%)
  - 🔴 **HIGH RISK** (> 40%)
- **Automated Care Action Plan**: Generates custom clinical discharge protocols (medication reconciliation, 48-hour post-discharge follow-up call, home health referrals, and primary care scheduling).

### 5. 📁 Batch Cohort Triage & CSV Export
- Interactive filtering of patient cohorts by risk score and clinical department.
- Triage table for hospital discharge planning teams with one-click **CSV Report Download**.

### 6. ⚡ PySpark Distributed Pipeline Architecture
- Full documentation and syntax-highlighted code viewer of the PySpark pipeline.
- Architecture comparison between distributed Spark ML and Scikit-Learn local inference.

---

## 📊 Model Evaluation Results

Evaluated on held-out test partitions (stratified split, positive class: readmitted `<30` days):

| Model Architecture | Threshold | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Weighted Decision Tree (Required)** | **0.45** | **0.6474** | **66.62%** | **16.49%** | **0.2566** | **60.57%** |
| **Random Forest Benchmark** | **0.45** | **0.6631** | **75.74%** | **15.31%** | **0.2543** | **53.84%** |
| **Gradient Boosting Benchmark** | **0.45** | **0.6631** | **69.26%** | **15.65%** | **0.2563** | **57.42%** |
| **Extra Trees Benchmark** | **0.45** | **0.6523** | **81.77%** | **13.82%** | **0.2358** | **46.85%** |
| **Logistic Regression Baseline** | **0.45** | **0.6607** | **68.52%** | **16.02%** | **0.2591** | **58.78%** |

> **Clinical Note on Metric Selection:** In clinical risk stratification with an 11.3% positive prevalence, high **Recall (Sensitivity)** and **ROC-AUC** are prioritized over raw accuracy. Standard unweighted models yield 88% accuracy by predicting "No Readmission" for everyone, missing 100% of high-risk patients. Our weighted model achieves **>66% Recall**, capturing the majority of readmissions for early clinical intervention.

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    A["Raw Dataset: UCI 130-US Hospitals (101,766 Encounters)"] --> B["Data Cleaning & Preprocessing (src/preprocess.py)"]
    B --> C1["Handle Nulls in Admission & Discharge IDs"]
    B --> C2["ICD-9 Diagnosis Grouping (Circulatory, Respiratory, Diabetes, etc.)"]
    B --> C3["Filter Deceased Patients (Discharge IDs 11, 19, 20, 21)"]
    B --> C4["Feature Engineering (Prior Inpatient Visits, Polypharmacy, HbA1c)"]
    
    C1 & C2 & C3 & C4 --> D["Clean Data (data/processed/diabetic_data_clean.parquet)"]
    
    D --> E1["PySpark Pipeline (src/pyspark_pipeline.py)"]
    E1 --> F1["StringIndexer + OneHotEncoder + VectorAssembler"]
    F1 --> G1["Weighted DecisionTreeClassifier (Spark ML)"]
    
    D --> E2["Scikit-Learn Pipeline (src/model_trainer.py)"]
    E2 --> F2["ColumnTransformer + StandardScaler + OHE"]
    F2 --> G2["Decision Tree, Random Forest & Logistic Regression"]
    
    G1 & G2 --> H["Model Artifacts (models/*.joblib, models/metrics.json)"]
    H --> I["Streamlit Web Application (app.py)"]
    I --> J1["Executive Dashboard"]
    I --> J2["Interactive EDA Studio"]
    I --> J3["Model Training Arena"]
    I --> J4["Clinical Decision Support (CDS) Calculator"]
    I --> J5["Batch Cohort Triage"]
```

---

## 📂 Repository Directory Structure

```text
Hospital-Readmission-Project/
├── app.py                                         # Main Streamlit Dashboard Application
├── requirements.txt                               # Production Python dependencies
├── README.md                                      # Comprehensive Project Documentation
├── .gitignore                                     # Git ignore configuration
├── INT315_Hospital_Readmission_PySpark_Fixed.ipynb # Academic Reference Notebook
├── src/
│   ├── preprocess.py                              # Data ingestion, ICD-9 mapping & cleaning
│   ├── model_trainer.py                           # Scikit-Learn training, calibration & evaluation
│   └── pyspark_pipeline.py                        # INT315 PySpark SparkSession & ML pipeline
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv                      # Raw UCI 130-US hospitals dataset (101k rows)
│   └── processed/
│       ├── diabetic_data_clean.parquet            # Optimized clean dataset for ultra-fast load
│       ├── diabetic_data_sample.csv               # 5,000-row sample CSV
│       └── test_sample_1000.csv                   # Pre-stratified 1,000-patient triage cohort
└── models/
    ├── decision_tree.joblib                       # Pretrained Weighted Decision Tree Pipeline
    ├── random_forest.joblib                       # Pretrained Random Forest Pipeline
    ├── logistic_regression.joblib                 # Pretrained Logistic Regression Pipeline
    └── metrics.json                               # Precomputed evaluation curves and metrics
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/samad-00/Hospital-Readmission-Project.git
cd Hospital-Readmission-Project
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with the platform!

---

## ⚡ Running the PySpark Pipeline
To execute the distributed PySpark training job independently:
```bash
python src/pyspark_pipeline.py
```

---

## 🔬 Dataset Citation

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670.  
> Available at: [UCI Machine Learning Repository: Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us-hospitals+for-years+1999-2008)

---

## 👨‍💻 Author & Acknowledgements

Developed by **Samad** for **INT315 — Hospital Readmission Prediction (Project 23)**.  
For questions or collaborations, visit the [GitHub Repository](https://github.com/samad-00/Hospital-Readmission-Project).

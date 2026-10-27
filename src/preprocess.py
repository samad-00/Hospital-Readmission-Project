import os
import re
import numpy as np
import pandas as pd

ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "Unknown",
    7: "Trauma Center",
    8: "Not Mapped"
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to Home",
    2: "Transferred to Short-term Hospital",
    3: "Transferred to SNF",
    4: "Transferred to ICF",
    5: "Transferred to Inpatient Care",
    6: "Home with Home Health",
    7: "Left AMA",
    8: "Home IV Provider",
    11: "Expired",
    13: "Hospice / Home",
    14: "Hospice / Medical Facility",
    18: "Unknown",
    19: "Expired at Home",
    20: "Expired in Medical Facility",
    21: "Expired (Unknown Place)",
    22: "Transferred to Rehab",
    25: "Not Mapped"
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Hospital Transfer",
    5: "SNF Transfer",
    6: "Other Healthcare Transfer",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    11: "Normal Delivery",
    17: "Unknown",
    20: "Not Mapped"
}

TOP_SPECIALTIES = [
    "InternalMedicine",
    "Emergency/Trauma",
    "Family/GeneralPractice",
    "Cardiology",
    "Surgery-General",
    "Nephrology",
    "Orthopedics",
    "Orthopedics-Reconstructive",
    "Radiologist",
    "Pulmonology",
    "Psychiatry",
    "Urology",
    "ObstetricsandGynecology"
]

def map_icd9_category(val):
    """
    Standard ICD-9 code categorization for clinical research:
    - Circulatory: 390-459, 785
    - Respiratory: 460-519, 786
    - Digestive: 520-579, 787
    - Diabetes: 250.xx
    - Injury: 800-999
    - Musculoskeletal: 710-739
    - Genitourinary: 580-629, 788
    - Neoplasms: 140-239
    - Other: all other codes
    """
    if pd.isna(val) or str(val).strip() in ["?", "", "None", "nan"]:
        return "Unknown"
    val = str(val).strip()
    if val.startswith("V") or val.startswith("E"):
        return "Other"
    try:
        code = float(val)
        if (390 <= code <= 459) or code == 785:
            return "Circulatory"
        elif (460 <= code <= 519) or code == 786:
            return "Respiratory"
        elif (520 <= code <= 579) or code == 787:
            return "Digestive"
        elif int(code) == 250:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif (580 <= code <= 629) or code == 788:
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except (ValueError, TypeError):
        return "Other"

def clean_specialty(val):
    if pd.isna(val) or str(val).strip() in ["?", "", "None", "nan"]:
        return "Unknown/Missing"
    val = str(val).strip()
    if val in TOP_SPECIALTIES:
        return val
    return "Other Specialty"

def load_and_preprocess_data(csv_path="data/raw/diabetic_data.csv", exclude_expired=True):
    """
    Loads raw diabetic dataset, handles missing/null values according to project requirements,
    encodes clinical categories, engineers clinical risk features, and sets up binary 30-day target.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data file not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Handle missing/null values in admission & discharge fields explicitly (Project 23 Requirement)
    # Replace '?' and nulls with appropriate categories
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    df["admission_type"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Other/Unknown")
    df["discharge_disposition"] = df["discharge_disposition_id"].map(DISCHARGE_DISPOSITION_MAP).fillna("Other/Unknown")
    df["admission_source"] = df["admission_source_id"].map(ADMISSION_SOURCE_MAP).fillna("Other/Unknown")

    # Filter out expired patients if requested (deceased patients cannot be readmitted within 30 days)
    if exclude_expired:
        expired_ids = [11, 19, 20, 21]
        df = df[~df["discharge_disposition_id"].isin(expired_ids)].copy()

    # 2. Binary Target for 30-day readmission (Project 23 Requirement)
    # 1 = readmitted <30 days; 0 = readmitted >30 days or NO readmission
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    # Multi-class target preserved for exploratory analysis: 'NO', '>30', '<30'
    df["readmitted_multiclass"] = df["readmitted"].astype(str)

    # 3. Handle diagnosis encoding into clinical categories (Project 23 Requirement)
    df["primary_diagnosis"] = df["diag_1"].apply(map_icd9_category)
    df["secondary_diagnosis"] = df["diag_2"].apply(map_icd9_category)
    df["additional_diagnosis"] = df["diag_3"].apply(map_icd9_category)

    # 4. Department / Medical Specialty encoding
    df["department"] = df["medical_specialty"].apply(clean_specialty)

    # 5. Clean Race and Gender
    df["race"] = df["race"].replace("?", "Unknown").fillna("Unknown")
    df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")

    # 6. Numeric features validation and cleaning
    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 7. Clinical Risk Feature Engineering
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["has_prior_emergency"] = (df["number_emergency"] > 0).astype(int)
    df["polypharmacy"] = (df["num_medications"] >= 16).astype(int)
    df["medication_changed"] = (df["change"] == "Ch").astype(int)
    df["on_diabetes_med"] = (df["diabetesMed"] == "Yes").astype(int)
    df["high_a1c"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

    # Clean A1Cresult & max_glu_serum
    df["A1Cresult"] = df["A1Cresult"].fillna("None").replace("?", "None")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None").replace("?", "None")

    return df

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    print("Preprocessing raw hospital data...")
    clean_df = load_and_preprocess_data()
    out_parquet = "data/processed/diabetic_data_clean.parquet"
    clean_df.to_parquet(out_parquet, index=False)
    out_sample_csv = "data/processed/diabetic_data_sample.csv"
    clean_df.head(5000).to_csv(out_sample_csv, index=False)
    print(f"Preprocessed data saved to {out_parquet} ({len(clean_df)} rows)")
    print(f"Sample test CSV saved to {out_sample_csv}")
    print("Target 30d readmission rate:", round(clean_df["readmitted_30d"].mean() * 100, 2), "%")

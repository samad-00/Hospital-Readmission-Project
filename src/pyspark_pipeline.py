"""
INT315 - Hospital Readmission Prediction with PySpark
Implementation based on Project 23 requirements:
- Load patient discharge records into Spark DataFrame
- Handle null values in admission and discharge fields
- Encode categorical data (diagnosis type, department, etc.)
- Build a Weighted Decision Tree classifier to predict readmission likelihood (<30 days)
- Evaluate model using accuracy, ROC-AUC, and PR-AUC
- Generate threshold tuning table
"""

import os
import sys

def run_spark_pipeline(data_path="data/raw/diabetic_data.csv"):
    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            StringIndexer,
            OneHotEncoder,
            VectorAssembler,
            Imputer
        )
        from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        from pyspark.ml.functions import vector_to_array
    except ImportError as e:
        print(f"PySpark not found or import error: {e}")
        return

    print("Initializing SparkSession...")
    spark = (
        SparkSession.builder
        .appName("INT315_Hospital_Readmission_PySpark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Reading patient records from {data_path}...")
    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(data_path)
    )
    print(f"Total encounters loaded: {raw_df.count()}, Columns: {len(raw_df.columns)}")

    # 1. Handle missing/null values in admission and discharge fields (Project 23 task)
    required_id_fields = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id"
    ]
    clean_df = raw_df
    for col in required_id_fields:
        clean_df = clean_df.withColumn(
            col,
            F.when(
                F.col(col).isNull() |
                (F.trim(F.col(col)) == "") |
                (F.trim(F.col(col)) == "?"),
                F.lit("Unknown")
            ).otherwise(F.trim(F.col(col)))
        )

    # Filter out expired patients (11, 19, 20, 21)
    clean_df = clean_df.filter(~F.col("discharge_disposition_id").isin(["11", "19", "20", "21"]))

    # 2. Binary target: 1 = readmitted <30 days, 0 = otherwise
    clean_df = clean_df.withColumn(
        "label",
        F.when(F.col("readmitted") == "<30", F.lit(1.0)).otherwise(F.lit(0.0))
    )

    # 3. ICD-9 diagnosis grouping
    def icd9_udf(code):
        if not code or code in ["?", "", "None"]:
            return "Unknown"
        if code.startswith("V") or code.startswith("E"):
            return "Other"
        try:
            val = float(code)
            if (390 <= val <= 459) or val == 785:
                return "Circulatory"
            elif (460 <= val <= 519) or val == 786:
                return "Respiratory"
            elif (520 <= val <= 579) or val == 787:
                return "Digestive"
            elif int(val) == 250:
                return "Diabetes"
            elif 800 <= val <= 999:
                return "Injury"
            elif 710 <= val <= 739:
                return "Musculoskeletal"
            elif (580 <= val <= 629) or val == 788:
                return "Genitourinary"
            elif 140 <= val <= 239:
                return "Neoplasms"
            else:
                return "Other"
        except:
            return "Other"

    spark_icd9 = F.udf(icd9_udf)
    clean_df = clean_df.withColumn("diag_1_category", spark_icd9(F.col("diag_1")))

    # 4. Define features
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
    for c in numeric_cols:
        clean_df = clean_df.withColumn(c, F.col(c).cast("double"))

    categorical_cols = [
        "age", "gender", "race", "admission_type_id",
        "discharge_disposition_id", "admission_source_id",
        "diag_1_category", "max_glu_serum", "A1Cresult",
        "metformin", "insulin", "change", "diabetesMed"
    ]

    for c in categorical_cols:
        clean_df = clean_df.withColumn(
            c,
            F.when(F.col(c).isNull() | (F.trim(F.col(c)) == "?"), F.lit("Unknown"))
            .otherwise(F.trim(F.col(c)))
        )

    # 5. Train/Val/Test Split
    train_df, val_df, test_df = clean_df.randomSplit([0.7, 0.15, 0.15], seed=42)

    # Calculate class weight for positive class
    majority = train_df.filter(F.col("label") == 0.0).count()
    minority = train_df.filter(F.col("label") == 1.0).count()
    pos_weight = float(majority) / max(minority, 1)
    print(f"Train size: {train_df.count()}, Positive weight: {pos_weight:.3f}")

    train_weighted = train_df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )

    # 6. Build Spark ML Pipeline
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoded_cols = [f"{c}_vec" for c in categorical_cols]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_cols],
        outputCols=encoded_cols,
        handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=numeric_cols + encoded_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    # 7. Decision Tree Classifier (Required model)
    dt = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        weightCol="class_weight",
        maxDepth=6,
        maxBins=64,
        seed=42
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, dt])
    print("Fitting PySpark Decision Tree Pipeline...")
    model = pipeline.fit(train_weighted)

    # 8. Evaluation on Test Set
    test_pred = model.transform(test_df)
    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR"
    )

    roc_auc = evaluator_roc.evaluate(test_pred)
    pr_auc = evaluator_pr.evaluate(test_pred)

    print("\n========== INT315 PYSPARK MODEL RESULTS ==========")
    print(f"Decision Tree ROC-AUC: {roc_auc:.4f}")
    print(f"Decision Tree PR-AUC:  {pr_auc:.4f}")
    print("==================================================\n")

    spark.stop()
    return roc_auc, pr_auc

if __name__ == "__main__":
    run_spark_pipeline()

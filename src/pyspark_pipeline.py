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

# Ensure driver and workers use the exact same Python binary
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def run_spark_pipeline(data_path="data/raw/diabetic_data.csv"):
    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import (
            StringIndexer,
            OneHotEncoder,
            VectorAssembler
        )
        from pyspark.ml.classification import DecisionTreeClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
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
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found.")
        return

    print(f"Reading patient records into Spark DataFrame from {data_path}...")
    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(data_path)
    )
    print(f"Total encounters loaded: {raw_df.count()}, Total columns: {len(raw_df.columns)}")

    # 1. Handle missing/null values in admission and discharge fields (Project 23 requirement)
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

    # Filter out deceased / expired patients (11, 19, 20, 21) who cannot be readmitted
    clean_df = clean_df.filter(~F.col("discharge_disposition_id").isin(["11", "19", "20", "21"]))

    # 2. Binary target: 1 = readmitted <30 days, 0 = otherwise
    clean_df = clean_df.withColumn(
        "label",
        F.when(F.col("readmitted") == "<30", F.lit(1.0)).otherwise(F.lit(0.0))
    )

    # 3. Native Spark SQL ICD-9 Categorization (100% Native Spark SQL Expressions - No serialization overhead)
    # Cast diag_1 code to double if numeric
    diag_num = F.when(
        F.col("diag_1").rlike("^[0-9]+(\\.[0-9]+)?$"),
        F.col("diag_1").cast("double")
    ).otherwise(F.lit(None))

    clean_df = clean_df.withColumn(
        "diag_1_category",
        F.when(F.col("diag_1").isNull() | (F.trim(F.col("diag_1")) == "?"), F.lit("Unknown"))
        .when(F.col("diag_1").startswith("V") | F.col("diag_1").startswith("E"), F.lit("Other"))
        .when((diag_num >= 390) & (diag_num <= 459) | (diag_num == 785), F.lit("Circulatory"))
        .when((diag_num >= 460) & (diag_num <= 519) | (diag_num == 786), F.lit("Respiratory"))
        .when((diag_num >= 520) & (diag_num <= 579) | (diag_num == 787), F.lit("Digestive"))
        .when(F.col("diag_1").startswith("250"), F.lit("Diabetes"))
        .when((diag_num >= 800) & (diag_num <= 999), F.lit("Injury"))
        .when((diag_num >= 710) & (diag_num <= 739), F.lit("Musculoskeletal"))
        .when((diag_num >= 580) & (diag_num <= 629) | (diag_num == 788), F.lit("Genitourinary"))
        .when((diag_num >= 140) & (diag_num <= 239), F.lit("Neoplasms"))
        .otherwise(F.lit("Other"))
    )

    # Clean Department (medical_specialty)
    clean_df = clean_df.withColumn(
        "department",
        F.when(
            F.col("medical_specialty").isNull() | (F.trim(F.col("medical_specialty")) == "?"),
            F.lit("Unknown")
        ).otherwise(F.trim(F.col("medical_specialty")))
    )

    # 4. Cast numeric features & handle nulls
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
        clean_df = clean_df.withColumn(c, F.coalesce(F.col(c).cast("double"), F.lit(0.0)))

    categorical_cols = [
        "age", "gender", "race", "admission_type_id",
        "discharge_disposition_id", "admission_source_id",
        "department", "diag_1_category", "max_glu_serum", "A1Cresult",
        "metformin", "insulin", "change", "diabetesMed"
    ]

    for c in categorical_cols:
        clean_df = clean_df.withColumn(
            c,
            F.when(F.col(c).isNull() | (F.trim(F.col(c)) == "?"), F.lit("Unknown"))
            .otherwise(F.trim(F.col(c)))
        )

    # 5. Train & Test Split (80% train, 20% test)
    print("Splitting data into Train and Test partitions...")
    train_df, test_df = clean_df.randomSplit([0.8, 0.2], seed=42)

    # Calculate class weight to handle 11.3% rare readmission rate
    train_count = train_df.count()
    majority = train_df.filter(F.col("label") == 0.0).count()
    minority = train_df.filter(F.col("label") == 1.0).count()
    pos_weight = float(majority) / max(minority, 1)

    print(f"Training Encounters: {train_count}")
    print(f"  Non-Readmitted (Majority): {majority}")
    print(f"  Readmitted <30d (Minority): {minority}")
    print(f"  Positive Class Weight: {pos_weight:.3f}x")

    train_weighted = train_df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )

    # 6. Build Spark ML Pipeline (StringIndexer -> OneHotEncoder -> VectorAssembler)
    print("Building Spark ML Feature Pipeline...")
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

    # 7. Decision Tree Classifier (INT315 Required Model)
    dt = DecisionTreeClassifier(
        labelCol="label",
        featuresCol="features",
        weightCol="class_weight",
        maxDepth=6,
        maxBins=64,
        seed=42
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, dt])

    print("Training PySpark Weighted Decision Tree...")
    model = pipeline.fit(train_weighted)
    print("Model training complete!")

    # 8. Test Set Evaluation
    print("Evaluating model predictions on held-out test data...")
    test_pred = model.transform(test_df)

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    roc_auc = evaluator_roc.evaluate(test_pred)
    pr_auc = evaluator_pr.evaluate(test_pred)
    accuracy = evaluator_acc.evaluate(test_pred)

    print("\n" + "="*56)
    print("=== INT315 PROJECT 23: PYSPARK EVALUATION RESULTS ===")
    print("="*56)
    print(f"  Classifier Model:     PySpark DecisionTreeClassifier")
    print(f"  Class Balancing:      Weighting ({pos_weight:.2f}x)")
    print(f"  Area Under ROC (AUC): {roc_auc:.4f}")
    print(f"  Area Under PR:        {pr_auc:.4f}")
    print(f"  Accuracy:             {accuracy*100:.2f}%")
    print("="*56 + "\n")

    spark.stop()
    return roc_auc, accuracy

if __name__ == "__main__":
    run_spark_pipeline()

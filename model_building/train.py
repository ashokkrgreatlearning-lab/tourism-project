
# =====================================================
# TOURISM PREDICTION - XGBOOST + MLFLOW
# =====================================================

# STEP 1: Import Libraries
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# =====================================================
# STEP 2: Define Project Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEPLOYMENT_DIR = os.path.join(PROJECT_DIR, "deployment")
MLFLOW_DIR = os.path.join(PROJECT_DIR, "mlruns")

MODEL_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "best_tourism_model_v1.joblib"
)

THRESHOLD_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "model_config.json"
)

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

print("Project Directory:", PROJECT_DIR)
print("Deployment Directory:", DEPLOYMENT_DIR)

# =====================================================
# STEP 3: Load Prepared Datasets
# =====================================================

required_files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for filename in required_files:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required dataset missing: {path}. "
            "Please execute prep.py first."
        )

Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))

ytrain = pd.read_csv(
    os.path.join(DATA_DIR, "ytrain.csv")
).squeeze("columns")

ytest = pd.read_csv(
    os.path.join(DATA_DIR, "ytest.csv")
).squeeze("columns")

print("\nDatasets loaded successfully!")
print("Xtrain:", Xtrain.shape)
print("Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape)
print("ytest:", ytest.shape)

# =====================================================
# STEP 4: Define Feature Groups
# =====================================================

numeric_cols = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

required_cols = numeric_cols + categorical_cols

missing_cols = [
    col for col in required_cols
    if col not in Xtrain.columns or col not in Xtest.columns
]

if missing_cols:
    raise ValueError(
        f"Missing feature columns: {missing_cols}"
    )

# Ensure numerical columns contain numeric values
for col in numeric_cols:
    Xtrain[col] = pd.to_numeric(
        Xtrain[col], errors="coerce"
    )
    Xtest[col] = pd.to_numeric(
        Xtest[col], errors="coerce"
    )

# Validate binary target
ytrain = pd.to_numeric(ytrain, errors="raise").astype(int)
ytest = pd.to_numeric(ytest, errors="raise").astype(int)

if set(ytrain.unique()) != {0, 1}:
    raise ValueError(
        "Training target must contain binary classes 0 and 1."
    )

if not set(ytest.unique()).issubset({0, 1}):
    raise ValueError(
        "Testing target must contain only 0 and 1."
    )

# =====================================================
# STEP 5: Create Preprocessing Pipelines
# =====================================================

numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
)

preprocessor = make_column_transformer(
    (numeric_pipeline, numeric_cols),
    (categorical_pipeline, categorical_cols),
    remainder="drop"
)

# =====================================================
# STEP 6: Handle Class Imbalance
# =====================================================

class_counts = ytrain.value_counts()

negative_count = class_counts[0]
positive_count = class_counts[1]

class_weight = negative_count / positive_count

print("\nClass Distribution:")
print(class_counts)

print("Scale Positive Weight:", round(class_weight, 4))

# =====================================================
# STEP 7: Create XGBoost Classifier
# =====================================================

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1,
    tree_method="hist"
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# =====================================================
# STEP 8: Define Hyperparameter Grid
# =====================================================

param_grid = {
    "xgbclassifier__n_estimators": [75, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.5, 0.6],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.5, 0.6]
}

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =====================================================
# STEP 9: Configure MLflow
# =====================================================

# Store experiments in Google Drive
tracking_uri = "file://" + MLFLOW_DIR

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(
    "Tourism_XGBoost_Experiment"
)

print("\nMLflow Tracking URI:")
print(mlflow.get_tracking_uri())

# =====================================================
# STEP 10: Train Model with MLflow
# =====================================================

classification_threshold = 0.50

with mlflow.start_run(
    run_name="Optimized_XGB_Model_PROD_Standalone"
):

    mlflow.log_param(
        "classification_threshold",
        classification_threshold
    )

    mlflow.log_param(
        "scale_pos_weight",
        float(class_weight)
    )

    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "roc_auc")

    print("\nStarting GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        n_jobs=1,
        scoring="roc_auc",
        verbose=1,
        error_score="raise",
        refit=True
    )

    grid_search.fit(Xtrain, ytrain)

    print("\nGridSearchCV completed!")

    # =================================================
    # STEP 11: Log Hyperparameter Trials
    # =================================================

    results = grid_search.cv_results_

    for i, params in enumerate(results["params"]):

        with mlflow.start_run(
            run_name=f"Trial_{i+1}",
            nested=True
        ):

            mlflow.log_params(params)

            mlflow.log_metrics({
                "mean_cv_roc_auc": float(
                    results["mean_test_score"][i]
                ),
                "std_cv_roc_auc": float(
                    results["std_test_score"][i]
                )
            })

    # =================================================
    # STEP 12: Select Best Model
    # =================================================

    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric(
        "best_cv_roc_auc",
        float(grid_search.best_score_)
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest Cross Validation ROC-AUC:",
        round(grid_search.best_score_, 4)
    )

    # =================================================
    # STEP 13: Generate Predictions
    # =================================================

    y_prob_train = best_model.predict_proba(Xtrain)[:, 1]
    y_prob_test = best_model.predict_proba(Xtest)[:, 1]

    y_pred_train = (
        y_prob_train >= classification_threshold
    ).astype(int)

    y_pred_test = (
        y_prob_test >= classification_threshold
    ).astype(int)

    # =================================================
    # STEP 14: Evaluate Model
    # =================================================

    metrics = {
        "train_accuracy": accuracy_score(
            ytrain, y_pred_train
        ),
        "train_precision": precision_score(
            ytrain, y_pred_train, zero_division=0
        ),
        "train_recall": recall_score(
            ytrain, y_pred_train, zero_division=0
        ),
        "train_f1_score": f1_score(
            ytrain, y_pred_train, zero_division=0
        ),
        "train_roc_auc": roc_auc_score(
            ytrain, y_prob_train
        ),
        "test_accuracy": accuracy_score(
            ytest, y_pred_test
        ),
        "test_precision": precision_score(
            ytest, y_pred_test, zero_division=0
        ),
        "test_recall": recall_score(
            ytest, y_pred_test, zero_division=0
        ),
        "test_f1_score": f1_score(
            ytest, y_pred_test, zero_division=0
        ),
        "test_roc_auc": roc_auc_score(
            ytest, y_prob_test
        )
    }

    print("\nModel Evaluation Metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            ytest,
            y_pred_test,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

    mlflow.log_metrics({
        key: float(value)
        for key, value in metrics.items()
    })

    # =================================================
    # STEP 15: Save Best Model
    # =================================================

    joblib.dump(
        best_model,
        MODEL_PATH,
        compress=3
    )

    # Save classification threshold separately
    model_config = {
        "classification_threshold": classification_threshold,
        "model_name": "best_tourism_model_v1.joblib",
        "target_column": "ProdTaken"
    }

    with open(THRESHOLD_PATH, "w") as f:
        json.dump(model_config, f, indent=4)

    # =================================================
    # STEP 16: Log Model Artifacts
    # =================================================

    mlflow.log_artifact(
        MODEL_PATH,
        artifact_path="model"
    )

    mlflow.log_artifact(
        THRESHOLD_PATH,
        artifact_path="model"
    )

    print("\nModel and configuration logged to MLflow!")

# =====================================================
# STEP 17: Verify Saved Model
# =====================================================

print("\nVerifying saved model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model saving failed: {MODEL_PATH}"
    )

model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print("Model exists: True")
print("Model size:", round(model_size, 2), "MB")

loaded_model = joblib.load(MODEL_PATH)

sample_probabilities = loaded_model.predict_proba(
    Xtest.head(5)
)[:, 1]

sample_predictions = (
    sample_probabilities >= classification_threshold
).astype(int)

print("\nModel loaded successfully!")
print("Sample probabilities:", sample_probabilities)
print("Sample predictions:", sample_predictions)

print("\nTraining pipeline completed successfully!")


import os
import pandas as pd

# ==========================================
# Step 1: Define Project Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

RAW_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "tourism.csv"
)

# ==========================================
# Step 2: Verify Dataset Exists
# ==========================================

if not os.path.isfile(RAW_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {RAW_PATH}"
    )

print("Dataset found:", RAW_PATH)

# ==========================================
# Step 3: Load Raw Dataset
# ==========================================

df = pd.read_csv(RAW_PATH)

print("\nDataset loaded successfully!")

# ==========================================
# Step 4: Validate Expected Columns
# ==========================================

expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

missing_columns = [
    col for col in expected_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

print("\nAll required columns validated!")

# ==========================================
# Step 5: Dataset Information
# ==========================================

print("\nDataset registered successfully!")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nDataset Columns:")
print(df.columns.tolist())

# ==========================================
# Step 6: Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Step 7: Target Variable Summary
# ==========================================

print("\nTarget Variable: ProdTaken")

print(df["ProdTaken"].value_counts(dropna=False))

print("\nTarget Distribution (%):")

print(
    df["ProdTaken"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

# ==========================================
# Step 8: Preview Dataset
# ==========================================

print("\nFirst Five Rows:")

print(df.head())

print("\nDataset Registration Completed!")

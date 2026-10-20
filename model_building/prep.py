
# ==========================================
# 1. Import Required Libraries
# ==========================================
import os
import pandas as pd

from sklearn.model_selection import train_test_split

# ==========================================
# 2. Define Project Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "tourism.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 3. Verify and Load Dataset
# ==========================================
if not os.path.isfile(RAW_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {RAW_PATH}"
    )

df = pd.read_csv(RAW_PATH)

print("Step 1: Dataset loaded successfully!")
print("Original shape:", df.shape)

# ==========================================
# 4. Let us now Remove Unnecessary Columns
# ==========================================
print("\nStep 2: Performing data cleaning...")

cols_to_drop = ["CustomerID", "Unnamed: 0"]

df = df.drop(
    columns=cols_to_drop,
    errors="ignore"
)

# ==========================================
# 5. Standardize Categorical Values
# ==========================================
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female"
    })

if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({
        "Single": "Unmarried"
    })

# ==========================================
# 6. Remove Duplicate Records
# ==========================================
duplicates = df.duplicated().sum()

print("Duplicate rows found:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()

# ==========================================
# 7. Validate Target Variable
# ==========================================
target_col = "ProdTaken"

if target_col not in df.columns:
    raise ValueError(
        f"Target column '{target_col}' not found!"
    )

# Remove rows with missing target values
df = df.dropna(subset=[target_col])

# Validate target classes
if df[target_col].nunique() < 2:
    raise ValueError(
        "Target must contain at least two classes."
    )

print("\nTarget distribution:")
print(df[target_col].value_counts())

# ==========================================
# 8. Separate Features and Target
# ==========================================
X = df.drop(columns=[target_col])
y = df[target_col]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# ==========================================
# 9. Train-Test Split
# ==========================================
print("\nStep 3: Splitting dataset...")

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 10. Save Processed Datasets
# ==========================================
Xtrain.to_csv(
    os.path.join(DATA_DIR, "Xtrain.csv"),
    index=False
)

Xtest.to_csv(
    os.path.join(DATA_DIR, "Xtest.csv"),
    index=False
)

ytrain.to_csv(
    os.path.join(DATA_DIR, "ytrain.csv"),
    index=False
)

ytest.to_csv(
    os.path.join(DATA_DIR, "ytest.csv"),
    index=False
)

# ==========================================
# 11. Display Final Results
# ==========================================
print("\nData preparation completed successfully!")

print("Training features:", Xtrain.shape)
print("Testing features:", Xtest.shape)

print("Training target:", ytrain.shape)
print("Testing target:", ytest.shape)

print("\nProcessed files saved in:", DATA_DIR)

print("\nGenerated Files:")
for filename in [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]:
    path = os.path.join(DATA_DIR, filename)
    print(filename, "Exists:", os.path.isfile(path))


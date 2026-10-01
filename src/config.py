from pathlib import Path


# Project and workspace paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


# Dataset
DATASET_PATH = WORKSPACE_ROOT / "datasets" / "telecom_customer" / "teleCust1000t.csv"



# Output directory
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# Data split
TEST_SIZE = 0.20
RANDOM_STATE = 4


# KNN
INITIAL_K = 3
MAX_K = 100
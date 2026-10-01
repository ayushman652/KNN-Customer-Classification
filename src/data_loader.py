import pandas as pd

from src.config import DATASET_PATH


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def display_dataset_info(df):
    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["custcat"].value_counts())
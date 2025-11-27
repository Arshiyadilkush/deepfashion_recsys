"""
loader.py — generic data loading utilities.
Handles reading DeepFashion metadata, images, and annotation files.
"""
import pandas as pd
from pathlib import Path

def load_deepfashion_index(csv_path: str) -> pd.DataFrame:
    """Load the cleaned DeepFashion index CSV."""
    df = pd.read_csv(csv_path)
    return df

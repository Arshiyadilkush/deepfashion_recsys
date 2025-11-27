"""
indexing.py — builds image → category/item/attribute mapping from annotation files.
"""
import pandas as pd
from pathlib import Path

def build_index(root: Path) -> pd.DataFrame:
    """Parse annotation files and return unified DataFrame."""
    split_df = pd.read_csv(root / "Eval" / "list_eval_partition.txt", 
                           sep=r"\s+", header=None, skiprows=2,
                           names=["rel_path","item_id","split"])
    split_df["image_path"] = split_df["rel_path"].apply(lambda x: str(root / x))
    return split_df

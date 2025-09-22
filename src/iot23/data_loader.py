import io
import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import (
    candidate_label_columns,
    detect_delimiter_from_bytes,
    simplify_family_value,
    take_head_bytes,
    map_to_prd_family,
    PRD_FAMILIES,
)


SUPPORTED_EXTS = {".csv", ".tsv", ".log", ".txt"}


def is_probably_table(name: str) -> bool:
    name_lower = name.lower()
    if not any(name_lower.endswith(ext) for ext in SUPPORTED_EXTS):
        return False
    # Prefer labeled connection logs and csvs
    return (
        "labeled" in name_lower
        or "conn" in name_lower
        or name_lower.endswith(".csv")
        or name_lower.endswith(".tsv")
    )


def _iter_files_in_zip(zip_path: Path) -> Iterable[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir() and is_probably_table(info.filename):
                yield info.filename


def _iter_files_in_folder(folder: Path) -> Iterable[Path]:
    for p in folder.rglob("*"):
        if p.is_file() and is_probably_table(p.name):
            yield p


def list_candidate_tables(dataset_path: str) -> List[str]:
    p = Path(dataset_path)
    items: List[str] = []
    if p.is_file() and p.suffix.lower() == ".zip":
        items = list(_iter_files_in_zip(p))
    elif p.is_dir():
        items = [str(x) for x in _iter_files_in_folder(p)]
    return items


def _read_member_from_zip(zip_path: Path, member: str, nrows: Optional[int]) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member, "r") as bf:
                head = take_head_bytes(bf, 131072)
                sep = detect_delimiter_from_bytes(head)
                # Reset for pandas
                bf.seek(0)
                # Zeek logs often start with '#fields' comments; tell pandas to ignore
                df = pd.read_csv(
                    bf,
                    sep=sep if sep else None,
                    engine="python" if sep is None else "c",
                    comment="#",
                    na_values=["-", "N/A", "nan", "NaN", "?"],
                    nrows=nrows,
                    low_memory=False,
                    encoding_errors="ignore",
                )
                df = _fix_merged_label_columns(df)
                return df
    except Exception:
        return None


def _read_file_from_disk(path: Path, nrows: Optional[int]) -> Optional[pd.DataFrame]:
    try:
        # Try auto-detect separator with python engine
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            na_values=["-", "N/A", "nan", "NaN", "?"],
            nrows=nrows,
            low_memory=False,
            encoding_errors="ignore",
        )
        df = _fix_merged_label_columns(df)
        return df
    except Exception:
        return None


def _fix_merged_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle IoT-23 CSV quirk where last header packs 'tunnel_parents   label   detailed-label'."""
    try:
        merged_cols = [c for c in df.columns if ("label" in c.lower() and "tunnel_parents" in c.lower() and "  " in c)]
        for mc in merged_cols:
            parts = [p.strip() for p in mc.split() if p.strip()]
            if len(parts) < 2:
                continue
            expanded = df[mc].astype(str).str.split(r"\s+", n=len(parts) - 1, expand=True)
            # Ensure correct number of columns
            while expanded.shape[1] < len(parts):
                expanded[len(parts) - 1] = np.nan
            expanded.columns = parts[: expanded.shape[1]]
            df = df.drop(columns=[mc]).join(expanded)
        return df
    except Exception:
        return df


def _choose_label_column(df: pd.DataFrame, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    cands = candidate_label_columns(list(df.columns))
    for c in cands:
        if c in df.columns:
            return c
    # Try to find something that looks like a label (short list of uniques)
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 50:
            return c
    return None


def load_sample(
    dataset_path: str,
    max_files: int = 8,
    rows_per_file: int = 60000,
    preferred_label: Optional[str] = None,
    include_benign: bool = True,
    task: str = "family",  # 'family' or 'binary'
    restrict_prd_families: bool = True,
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """Load a sampled dataframe from IoT-23.

    Args:
        dataset_path: Path to IoT-23 dataset (zip file or directory)
        max_files: Maximum number of files to process
        rows_per_file: Maximum rows to sample from each file
        preferred_label: Preferred label column name
        include_benign: Whether to include benign samples
        task: Task type ('family' or 'binary')
        restrict_prd_families: Whether to restrict to PRD families only

    Returns:
        Tuple of (dataframe, target_column, loaded_files)
        
    Raises:
        ValueError: If dataset path is invalid or no data is loaded
        FileNotFoundError: If dataset path doesn't exist
    """
    # Validate inputs
    if not dataset_path or not isinstance(dataset_path, str):
        raise ValueError("Dataset path must be a non-empty string")
    
    if max_files < 1:
        raise ValueError("max_files must be at least 1")
    
    if rows_per_file < 1:
        raise ValueError("rows_per_file must be at least 1")
    
    if task not in ["family", "binary"]:
        raise ValueError("task must be either 'family' or 'binary'")
    
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    files: List[str] = []
    dfs: List[pd.DataFrame] = []
    loaded: List[str] = []

    if p.is_file() and p.suffix.lower() == ".zip":
        files = list(_iter_files_in_zip(p))
        # Favor 'labeled' files first
        files.sort(key=lambda x: ("labeled" not in x.lower(), x.lower()))
        for member in files[:max_files]:
            df = _read_member_from_zip(p, member, nrows=rows_per_file)
            if df is None or df.empty:
                continue
            df["__source_file__"] = member
            dfs.append(df)
            loaded.append(member)
    elif p.is_dir():
        disk_files = [Path(x) for x in list(_iter_files_in_folder(p))]
        disk_files.sort(key=lambda x: ("labeled" not in x.name.lower(), str(x).lower()))
        for fp in disk_files[:max_files]:
            df = _read_file_from_disk(fp, nrows=rows_per_file)
            if df is None or df.empty:
                continue
            df["__source_file__"] = str(fp)
            dfs.append(df)
            loaded.append(str(fp))

    if not dfs:
        raise ValueError(f"No valid data files found in {dataset_path}. Please check the path and file formats.")

    # Align columns and concatenate
    try:
        base_cols = set(dfs[0].columns)
        all_cols = list(base_cols.union(*[set(d.columns) for d in dfs]))
        dfs = [d.reindex(columns=all_cols) for d in dfs]
        df_all = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        raise ValueError(f"Failed to concatenate dataframes: {str(e)}")

    if df_all.empty:
        raise ValueError("No data loaded after concatenation. Check file contents and formats.")

    # Choose target
    target = _choose_label_column(df_all, preferred=preferred_label)

    if target is None:
        return df_all, None, loaded

    # Validate target column
    if target not in df_all.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Clean target for task
    try:
        if task == "binary":
            df_all[target] = df_all[target].astype(str)
            df_all[target] = np.where(df_all[target].str.contains("benign", case=False, na=False), "Benign", "Malicious")
            if not include_benign:
                df_all = df_all[df_all[target] != "Benign"]
        else:
            # family task: map to PRD families and/or simplify
            if restrict_prd_families:
                fam = df_all[target].map(map_to_prd_family)
                df_all[target] = fam
                df_all = df_all[df_all[target].isin(PRD_FAMILIES)]
                if not include_benign:
                    df_all = df_all[df_all[target] != "Benign"]
            else:
                df_all[target] = df_all[target].map(lambda x: simplify_family_value(str(x)))
                if not include_benign:
                    df_all = df_all[~df_all[target].str.contains("benign", case=False, na=False)]

        df_all = df_all.reset_index(drop=True)
        
        # Final validation
        if df_all.empty:
            raise ValueError("No data remaining after filtering. Try adjusting include_benign or restrict_prd_families settings.")
        
        if target in df_all.columns and df_all[target].nunique() < 2:
            raise ValueError(f"Only one class found in target column '{target}'. Try increasing sample size or adjusting filters.")
            
    except Exception as e:
        raise ValueError(f"Failed to process target column: {str(e)}")

    return df_all, target, loaded

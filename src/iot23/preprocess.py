from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Basic dtype-based selection
    # Drop known label-like columns that can sneak into features (IoT-23 specifics)
    label_like = {
        "label",
        "detailed-label",
        "detailed_label",
        "family",
        "malware",
        "threat",
        "class",
        "tunnel_parents",
    }
    # Also drop obvious IDs/high-cardinality fields and helper columns
    drop_cols = {"uid", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "__source_file__"}
    drop_all = {c for c in X.columns if c.lower() in label_like} | drop_cols
    if drop_all:
        X = X.drop(columns=[c for c in drop_all if c in X.columns])

    numeric_cols: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    bool_cols: List[str] = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]

    # Drop/high-cardinality identifiers that explode one-hot dimensions
    drop_cols = {"uid", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "__source_file__"}
    safe_cats_allowlist = {"proto", "service", "conn_state", "history"}
    max_unique_for_onehot = 100  # cap categories per column

    obj_cols: List[str] = [
        c for c in X.columns
        if c not in numeric_cols and c not in bool_cols and X[c].dtype == object and c not in drop_cols
    ]
    # Keep only low-cardinality categorical columns or allowlisted protocol fields
    categorical_cols: List[str] = []
    for c in obj_cols:
        try:
            nunique = X[c].nunique(dropna=True)
        except Exception:
            nunique = max_unique_for_onehot + 1
        if c in safe_cats_allowlist or nunique <= max_unique_for_onehot:
            categorical_cols.append(c)

    # Cast booleans to integers for models that need numeric
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype("float32")

    transformers = []
    if numeric_cols or bool_cols:
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols + bool_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                # Limit category explosion and bin infrequent values
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    max_categories=max_unique_for_onehot,
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical_cols,
            )
        )

    if not transformers:
        # Fallback to identity on an empty set – rare, but keeps pipeline happy
        transformers.append(("passthrough", "passthrough", []))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre

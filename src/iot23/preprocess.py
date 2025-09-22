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
    numeric_cols: List[str] = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    bool_cols: List[str] = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    categorical_cols: List[str] = [
        c for c in X.columns if c not in numeric_cols and c not in bool_cols and X[c].dtype == object
    ]

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
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical_cols,
            )
        )

    if not transformers:
        # Fallback to identity on an empty set – rare, but keeps pipeline happy
        transformers.append(("passthrough", "passthrough", []))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre


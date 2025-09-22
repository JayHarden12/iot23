from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional deps
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
except Exception:  # pragma: no cover - optional
    tf = None
    keras = None

# Removed XGBoost and SVM per requirements

from .preprocess import build_preprocessor, split_features_target


@dataclass
class TrainConfig:
    model_name: str = "RandomForest"  # 'RandomForest' | 'CNN1D' | 'LSTM'
    test_size: float = 0.2
    random_state: int = 42
    # Model hyperparameters
    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    # DL params
    epochs: int = 10
    batch_size: int = 256
    patience: int = 3
    learning_rate_dl: float = 0.001


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, architecture: str = "CNN1D", epochs: int = 10, batch_size: int = 256, patience: int = 3, random_state: int = 42, verbose: int = 0):
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.classes_ = None
        self.le_ = None

    def _build_model(self, input_dim: int, n_classes: int):  # pragma: no cover - depends on TF
        if keras is None:
            raise RuntimeError("TensorFlow/Keras not installed. Install 'tensorflow' to use CNN/LSTM models.")
        inputs = keras.Input(shape=(input_dim,), name="features")
        x = inputs
        if self.architecture.upper() == "CNN1D":
            x = keras.layers.Reshape((input_dim, 1))(x)
            x = keras.layers.Conv1D(64, 3, activation="relu")(x)
            x = keras.layers.Conv1D(64, 3, activation="relu")(x)
            x = keras.layers.GlobalMaxPooling1D()(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
        elif self.architecture.upper() == "LSTM":
            x = keras.layers.Reshape((input_dim, 1))(x)
            x = keras.layers.LSTM(64, return_sequences=False)(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
        else:
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):  # pragma: no cover - depends on TF
        if keras is None:
            raise RuntimeError("TensorFlow/Keras not installed. Install 'tensorflow' to use CNN/LSTM models.")
        X = self._ensure_array(X)
        self.le_ = LabelEncoder().fit(y)
        y_enc = self.le_.transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        self.model_ = self._build_model(X.shape[1], n_classes)
        # class weights (balanced)
        _, counts = np.unique(y_enc, return_counts=True)
        total = counts.sum()
        class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)]
        self.model_.fit(
            X.astype("float32"),
            y_enc.astype("int32"),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=self.verbose,
            class_weight=class_weight,
        )
        return self

    def predict_proba(self, X):  # pragma: no cover - depends on TF
        X = self._ensure_array(X)
        probs = self.model_.predict(X.astype("float32"), verbose=0)
        return probs

    def predict(self, X):  # pragma: no cover - depends on TF
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.le_.inverse_transform(idx)

    @staticmethod
    def _ensure_array(X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)


def build_model(cfg: TrainConfig) -> Pipeline:
    if cfg.model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    elif cfg.model_name in {"CNN1D", "LSTM"}:
        clf = KerasClassifierWrapper(
            architecture=cfg.model_name, 
            epochs=cfg.epochs, 
            batch_size=cfg.batch_size, 
            patience=cfg.patience, 
            random_state=cfg.random_state
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}. Supported models: RandomForest, CNN1D, LSTM")
    
    pipe = Pipeline(steps=[
        ("pre", "passthrough"),  # placeholder, set later once X known
        ("clf", clf),
    ])
    return pipe


def train_and_evaluate(df: pd.DataFrame, target: str, cfg: TrainConfig) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Train and evaluate a model on the given data.
    
    Args:
        df: Input dataframe
        target: Target column name
        cfg: Training configuration
        
    Returns:
        Tuple of (pipeline, metrics, confusion_matrix, feature_importance, classification_report)
        
    Raises:
        ValueError: If data is invalid or training fails
    """
    # Validate inputs
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or None")
    
    if not target or target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    if not isinstance(cfg, TrainConfig):
        raise ValueError("cfg must be a TrainConfig instance")
    
    X, y = split_features_target(df, target)
    
    if X.empty:
        raise ValueError("No features found in data")
    
    if len(pd.unique(y)) < 2:
        raise ValueError("Training requires at least 2 classes; load more data or relax PRD family filter.")
    # Drop singleton classes (cannot stratify with 1 sample)
    vc = y.value_counts()
    dropped_classes = [str(c) for c in vc[vc < 2].index.tolist()]
    if dropped_classes:
        df = df[~df[target].isin(dropped_classes)].copy()
        X, y = split_features_target(df, target)
        if len(pd.unique(y)) < 2:
            raise ValueError(
                "After removing singleton classes, only one class remains. Increase sample size, adjust PRD family restriction, or upload more files."
            )

    split_strategy = "stratified"
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
        )
    except ValueError as e:
        # Fallback to non-stratified split when stratification is not feasible
        split_strategy = "random"
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=None
            )
        except Exception as e:
            raise ValueError(f"Failed to split data: {str(e)}")

    try:
        pre = build_preprocessor(X_train.copy())
        pipe = build_model(cfg)
        pipe.set_params(pre=pre)
    except Exception as e:
        raise ValueError(f"Failed to build model pipeline: {str(e)}")

    try:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    except Exception as e:
        raise ValueError(f"Model training failed: {str(e)}")

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    metrics["split_strategy"] = split_strategy
    if dropped_classes:
        try:
            metrics["dropped_classes"] = ", ".join(sorted(set(dropped_classes)))
        except Exception:
            metrics["dropped_classes"] = str(dropped_classes)

    # Ensure both inputs are pandas objects before concatenation
    labels = sorted(
        pd.unique(
            pd.concat([
                y_test.reset_index(drop=True) if hasattr(y_test, "reset_index") else pd.Series(y_test),
                pd.Series(y_pred),
            ], ignore_index=True)
        )
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    # ROC-AUC / PR-AUC (macro, OVR)
    try:
        # Need probabilities and 2+ classes
        proba = pipe.predict_proba(X_test)
        classes = getattr(pipe.named_steps.get("clf"), "classes_", None)
        if proba is not None and classes is not None and len(np.unique(y_test)) > 1:
            y_true_bin = label_binarize(y_test, classes=classes)
            # Align proba columns to classes
            # proba shape (n, n_classes) assumed aligned
            metrics["roc_auc_macro_ovr"] = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
            # PR-AUC macro: average of per-class AP
            ap_per_class = []
            for i in range(y_true_bin.shape[1]):
                ap_per_class.append(average_precision_score(y_true_bin[:, i], proba[:, i]))
            if ap_per_class:
                metrics["pr_auc_macro_ovr"] = float(np.mean(ap_per_class))
    except Exception:
        pass

    # Feature importance (best-effort)
    feature_importance: Dict[str, float] = {}
    try:
        clf = pipe.named_steps.get("clf")
        pre = pipe.named_steps.get("pre")
        if hasattr(clf, "feature_importances_"):
            # Build feature names from preprocessor
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
            else:
                names = [f"f{i}" for i in range(len(clf.feature_importances_))]
            importances = clf.feature_importances_
            order = np.argsort(importances)[::-1]
            for idx in order[:50]:
                feature_importance[str(names[idx])] = float(importances[idx])
        elif hasattr(clf, "coef_"):
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
            else:
                names = [f"f{i}" for i in range(len(clf.coef_.ravel()))]
            coefs = np.abs(clf.coef_).ravel()
            order = np.argsort(coefs)[::-1]
            for idx in order[:50]:
                feature_importance[str(names[idx])] = float(coefs[idx])
        else:
            # Fallback: light permutation importance on a small sample
            Xs = X_test.sample(min(2000, len(X_test)), random_state=cfg.random_state)
            ys = y_test.loc[Xs.index]
            r = permutation_importance(pipe, Xs, ys, n_repeats=3, random_state=cfg.random_state, n_jobs=-1)
            if hasattr(pre, "get_feature_names_out"):
                names = pre.get_feature_names_out()
            else:
                names = [f"f{i}" for i in range(len(r.importances_mean))]
            order = np.argsort(r.importances_mean)[::-1]
            for idx in order[:50]:
                feature_importance[str(names[idx])] = float(r.importances_mean[idx])
    except Exception:
        pass

    # Per-class report
    try:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
    except Exception:
        report_df = pd.DataFrame()

    return pipe, metrics, cm_df, feature_importance, report_df

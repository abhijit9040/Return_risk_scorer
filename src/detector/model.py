"""LightGBM multi-class detector + SHAP reason codes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import CLASS_NAMES, METRICS_DIR, MODELS_DIR, RANDOM_STATE
from src.features import build_preprocessor, prepare_xy


class ReturnRiskDetector:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()
        self.pipeline: Pipeline | None = None
        self.category_abuse_rate: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self._shap_explainer: shap.TreeExplainer | None = None

    def fit(self, train_df: pd.DataFrame) -> "ReturnRiskDetector":
        X, y, meta = prepare_xy(train_df, fit=True)
        assert y is not None
        self.category_abuse_rate = meta["category_abuse_rate"]
        # Lock label order to CLASS_NAMES for stable metrics / confusion matrix
        self.label_encoder.fit(CLASS_NAMES)
        y_enc = self.label_encoder.transform(y)

        clf = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        self.pipeline = Pipeline(
            steps=[
                ("pre", build_preprocessor()),
                ("clf", clf),
            ]
        )
        self.pipeline.fit(X, y_enc)
        pre = self.pipeline.named_steps["pre"]
        num_names = list(pre.transformers_[0][2])
        cat_names = list(pre.named_transformers_["cat"].get_feature_names_out())
        self.feature_names_ = num_names + cat_names
        self._shap_explainer = None
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = prepare_xy(df, category_abuse_rate=self.category_abuse_rate, fit=False)
        assert self.pipeline is not None
        return self.pipeline.predict_proba(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        idx = proba.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        classes = list(self.label_encoder.classes_)
        out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=df.index)
        pred_idx = proba.argmax(axis=1)
        out["predicted_class"] = self.label_encoder.inverse_transform(pred_idx)
        # Risk score = 1 - P(Legitimate); higher = more abusive
        if "p_Legitimate" in out.columns:
            out["risk_score"] = 1.0 - out["p_Legitimate"]
        else:
            out["risk_score"] = 1.0 - proba.max(axis=1)
        out["confidence"] = proba.max(axis=1)
        return out

    def _transformed(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = prepare_xy(df, category_abuse_rate=self.category_abuse_rate, fit=False)
        assert self.pipeline is not None
        return self.pipeline.named_steps["pre"].transform(X)

    def shap_top_reasons(self, df: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
        """Return top-k SHAP reason codes per row (feature, direction, magnitude)."""
        assert self.pipeline is not None
        clf = self.pipeline.named_steps["clf"]
        Xt = self._transformed(df)
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(clf)

        shap_values = self._shap_explainer.shap_values(Xt)
        # LightGBM multiclass: list of arrays or 3D array depending on shap version
        if isinstance(shap_values, list):
            # pick predicted class contribution per row
            proba = clf.predict_proba(Xt)
            pred = proba.argmax(axis=1)
            per_row = []
            for i, cls_idx in enumerate(pred):
                vals = np.asarray(shap_values[cls_idx][i]).ravel()
                per_row.append(vals)
            matrix = np.vstack(per_row)
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                # (n, features, classes) or (classes, n, features)
                if arr.shape[0] == len(df) or arr.shape[0] == Xt.shape[0]:
                    proba = clf.predict_proba(Xt)
                    pred = proba.argmax(axis=1)
                    matrix = np.stack([arr[i, :, pred[i]] for i in range(len(pred))])
                else:
                    proba = clf.predict_proba(Xt)
                    pred = proba.argmax(axis=1)
                    matrix = np.stack([arr[pred[i], i, :] for i in range(len(pred))])
            else:
                matrix = arr

        reasons: list[list[dict[str, Any]]] = []
        names = self.feature_names_ or [f"f{i}" for i in range(matrix.shape[1])]
        for row in matrix:
            order = np.argsort(np.abs(row))[::-1][:top_k]
            reasons.append(
                [
                    {
                        "feature": names[j] if j < len(names) else f"f{j}",
                        "shap_value": float(row[j]),
                        "direction": "increases_risk" if row[j] > 0 else "decreases_risk",
                    }
                    for j in order
                ]
            )
        return reasons

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, Any]:
        y_true = test_df["abuse_type"].astype(str)
        scored = self.score_frame(test_df)
        y_pred = scored["predicted_class"]
        labels = list(self.label_encoder.classes_)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        metrics = {
            "labels": labels,
            "precision_per_class": dict(zip(labels, map(float, precision))),
            "recall_per_class": dict(zip(labels, map(float, recall))),
            "f1_per_class": dict(zip(labels, map(float, f1))),
            "support_per_class": dict(zip(labels, map(int, support))),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        (METRICS_DIR / "classification_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return metrics

    def save(self, path: Path | None = None) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or (MODELS_DIR / "detector.joblib")
        payload = {
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "category_abuse_rate": self.category_abuse_rate,
            "feature_names_": self.feature_names_,
        }
        joblib.dump(payload, path)
        meta = {
            "classes": list(self.label_encoder.classes_),
            "n_features": len(self.feature_names_),
            "category_abuse_rate": self.category_abuse_rate,
        }
        (MODELS_DIR / "detector_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ReturnRiskDetector":
        path = path or (MODELS_DIR / "detector.joblib")
        payload = joblib.load(path)
        obj = cls()
        obj.pipeline = payload["pipeline"]
        obj.label_encoder = payload["label_encoder"]
        obj.category_abuse_rate = payload["category_abuse_rate"]
        obj.feature_names_ = payload.get("feature_names_", [])
        return obj

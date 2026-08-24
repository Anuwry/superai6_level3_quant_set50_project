from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import Pipeline

from models.track_b_data import (
    AMBIGUOUS_LABEL,
    IRRELEVANT_LABELS,
    POLARITY_LABELS,
    target_aware_text,
)

PROBABILITY_COLUMNS = {
    "positive": "positive_probability",
    "neutral": "neutral_probability",
    "negative": "negative_probability",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Data frame is missing columns: {missing}")


def make_relevance_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=labels.index, dtype=float)
    result.loc[normalized.isin(POLARITY_LABELS)] = 1.0
    result.loc[normalized.isin(IRRELEVANT_LABELS)] = 0.0
    result.loc[normalized.eq(AMBIGUOUS_LABEL)] = np.nan
    unknown = normalized.loc[
        ~normalized.isin(POLARITY_LABELS | IRRELEVANT_LABELS | {AMBIGUOUS_LABEL})
    ]
    if not unknown.empty:
        raise ValueError(f"Unknown Track B labels: {sorted(unknown.unique())}")
    return result


def classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
    *,
    labels: tuple[str, ...],
) -> dict[str, object]:
    expected_values = list(expected)
    predicted_values = list(predicted)
    if not expected_values or len(expected_values) != len(predicted_values):
        raise ValueError(
            "Expected and predicted labels must have equal non-zero length"
        )
    return {
        "accuracy": float(accuracy_score(expected_values, predicted_values)),
        "macro_f1": float(
            f1_score(
                expected_values,
                predicted_values,
                labels=list(labels),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                expected_values,
                predicted_values,
                labels=list(labels),
                average="weighted",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(expected_values, predicted_values)),
        "confusion_matrix": confusion_matrix(
            expected_values,
            predicted_values,
            labels=list(labels),
        ).tolist(),
        "classification_report": classification_report(
            expected_values,
            predicted_values,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
    }


@dataclass(frozen=True)
class BaselineRuntime:
    fit_seconds: float
    predict_seconds: float


class LocalSentimentClassifier:
    def __init__(
        self,
        *,
        max_features: int = 60_000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (3, 5),
        random_seed: int = 42,
    ) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        self.runtime: BaselineRuntime | None = None
        self._fit_seconds = 0.0

    @staticmethod
    def _inputs(frame: pd.DataFrame) -> list[str]:
        _require_columns(frame, {"ticker", "text"})
        return [
            target_aware_text(ticker, text)
            for ticker, text in zip(frame["ticker"], frame["text"], strict=True)
        ]

    def fit(self, frame: pd.DataFrame) -> LocalSentimentClassifier:
        _require_columns(frame, {"label"})
        if not frame["label"].isin(POLARITY_LABELS).all():
            raise ValueError("Sentiment training data contains non-polarity labels")
        if frame["label"].nunique() < 2:
            raise ValueError("Sentiment training data requires at least two classes")
        started = time.perf_counter()
        self.pipeline.fit(self._inputs(frame), frame["label"].astype(str))
        self._fit_seconds = time.perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        started = time.perf_counter()
        probabilities = self.pipeline.predict_proba(self._inputs(frame))
        predict_seconds = time.perf_counter() - started
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        output = frame[
            [
                column
                for column in ("article_id", "date", "year", "ticker", "label")
                if column in frame
            ]
        ].copy()
        for label, column in PROBABILITY_COLUMNS.items():
            output[column] = (
                probabilities[:, classes.index(label)] if label in classes else 0.0
            )
        probability_names = list(PROBABILITY_COLUMNS.values())
        probability_values = output[probability_names].to_numpy(dtype=float)
        labels = list(PROBABILITY_COLUMNS)
        output["predicted_label"] = np.asarray(labels)[
            probability_values.argmax(axis=1)
        ]
        output["sentiment_score"] = (
            output["positive_probability"] - output["negative_probability"]
        )
        output["confidence"] = probability_values.max(axis=1)
        self.runtime = BaselineRuntime(self._fit_seconds, predict_seconds)
        return output


class LocalRelevanceClassifier:
    def __init__(
        self,
        *,
        max_features: int = 60_000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (3, 5),
        random_seed: int = 42,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Relevance threshold must be between zero and one")
        self.threshold = threshold
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        self.runtime: BaselineRuntime | None = None
        self._fit_seconds = 0.0

    @staticmethod
    def _inputs(frame: pd.DataFrame) -> list[str]:
        return LocalSentimentClassifier._inputs(frame)

    def fit(self, frame: pd.DataFrame) -> LocalRelevanceClassifier:
        _require_columns(frame, {"label"})
        targets = make_relevance_labels(frame["label"])
        included = targets.notna()
        if targets.loc[included].nunique() != 2:
            raise ValueError("Relevance training data requires both binary classes")
        started = time.perf_counter()
        self.pipeline.fit(
            self._inputs(frame.loc[included]),
            targets.loc[included].astype(int),
        )
        self._fit_seconds = time.perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        started = time.perf_counter()
        probabilities = self.pipeline.predict_proba(self._inputs(frame))
        predict_seconds = time.perf_counter() - started
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        relevant_probability = probabilities[:, classes.index(1)]
        output = frame[
            [
                column
                for column in ("article_id", "date", "year", "ticker", "label")
                if column in frame
            ]
        ].copy()
        output["relevant_probability"] = relevant_probability
        output["predicted_relevant"] = (relevant_probability >= self.threshold).astype(
            int
        )
        self.runtime = BaselineRuntime(self._fit_seconds, predict_seconds)
        return output

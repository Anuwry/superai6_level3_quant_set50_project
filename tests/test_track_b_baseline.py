import numpy as np
import pandas as pd

from models.track_b_baseline import (
    LocalRelevanceClassifier,
    LocalSentimentClassifier,
    classification_metrics,
    make_relevance_labels,
)


def _training_pairs() -> pd.DataFrame:
    rows = []
    examples = {
        "positive": ["กำไรเพิ่ม รายได้โต", "จ่ายปันผลสูง แนวโน้มดี"],
        "neutral": ["แจ้งประชุมผู้ถือหุ้น", "รายงานข้อมูลประจำปี"],
        "negative": ["ขาดทุนเพิ่ม รายได้ลด", "ผิดนัดชำระหนี้ แนวโน้มแย่"],
    }
    for label, texts in examples.items():
        for index, text in enumerate(texts):
            rows.append(
                {
                    "article_id": f"{label}-{index}",
                    "ticker": "AAA",
                    "text": text,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_local_sentiment_classifier_returns_normalized_three_class_probabilities():
    frame = _training_pairs()
    classifier = LocalSentimentClassifier(
        max_features=2_000,
        min_df=1,
        ngram_range=(2, 4),
    )

    classifier.fit(frame)
    prediction = classifier.predict(frame)

    assert prediction["predicted_label"].isin({"positive", "neutral", "negative"}).all()
    assert np.allclose(
        prediction[
            ["positive_probability", "neutral_probability", "negative_probability"]
        ].sum(axis=1),
        1.0,
    )
    assert prediction["sentiment_score"].between(-1.0, 1.0).all()
    assert prediction["confidence"].between(0.0, 1.0).all()


def test_relevance_labels_keep_ambiguous_out_of_binary_training():
    labels = pd.Series(
        ["positive", "neutral", "negative", "exclude", "not stock", "ambiguous"]
    )

    relevance = make_relevance_labels(labels)

    assert relevance.iloc[:5].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert np.isnan(relevance.iloc[5])


def test_local_relevance_classifier_returns_probability_and_boolean_decision():
    frame = pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3", "a4"],
            "ticker": ["AAA"] * 4,
            "text": [
                "AAA รายงานกำไรเพิ่ม",
                "AAA จ่ายปันผล",
                "บทความทั่วไปไม่เกี่ยวข้อง",
                "ข่าวต่างประเทศไม่กล่าวถึง AAA",
            ],
            "label": ["positive", "neutral", "exclude", "not stock"],
        }
    )
    classifier = LocalRelevanceClassifier(
        max_features=2_000,
        min_df=1,
        ngram_range=(2, 4),
    )

    classifier.fit(frame)
    prediction = classifier.predict(frame)

    assert prediction["relevant_probability"].between(0.0, 1.0).all()
    assert prediction["predicted_relevant"].isin([0, 1]).all()


def test_classification_metrics_include_paper_facing_scores():
    metrics = classification_metrics(
        ["positive", "neutral", "negative"],
        ["positive", "negative", "negative"],
        labels=("positive", "neutral", "negative"),
    )

    assert set(metrics) >= {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "confusion_matrix",
        "classification_report",
    }

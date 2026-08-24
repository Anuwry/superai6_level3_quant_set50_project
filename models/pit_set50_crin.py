from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROTOCOL_ID = "pit-set50-crin-2024-2025-v1"
MODEL_KEY = "pit_set50_crin"
MODEL_LABEL = "PIT-SET50-CRIN"
WINDOW = 20
CONSTITUENT_FEATURES = (
    "close_log_return_1d",
    "close_log_return_5d_scaled",
    "intraday_range",
    "log_volume_z20",
)
CONSTITUENT_LSTM_UNITS = 8
CONSTITUENT_HIDDEN_UNITS = 8
LEADER_HIDDEN_UNITS = 6
LEADER_CORRECTION_CAP = 0.25
LEARNING_RATE = 0.001
BOTTOM_AUXILIARY_WEIGHT = 0.20
MIN_ACTIVE_MEMBERS = 35


@dataclass(frozen=True)
class DirectionMetrics:
    observations: int
    balanced_accuracy: float
    direction_accuracy: float
    mcc: float
    tn: int
    fp: int
    fn: int
    tp: int
    predicted_up_share: float
    brier: float


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("Probabilities must be non-empty and finite")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def direction_metrics(y_true: np.ndarray, probability: np.ndarray) -> DirectionMetrics:
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

    truth = np.asarray(y_true, dtype=int).reshape(-1)
    score = np.asarray(probability, dtype=float).reshape(-1)
    if truth.shape != score.shape or len(truth) < 1:
        raise ValueError("Direction truth and probability must be non-empty and aligned")
    if not set(np.unique(truth)).issubset({0, 1}) or not np.isfinite(score).all():
        raise ValueError("Direction inputs are invalid")
    score = np.clip(score, 0.0, 1.0)
    prediction = score >= 0.5
    tn = int(np.sum((truth == 0) & (~prediction)))
    fp = int(np.sum((truth == 0) & prediction))
    fn = int(np.sum((truth == 1) & (~prediction)))
    tp = int(np.sum((truth == 1) & prediction))
    return DirectionMetrics(
        observations=len(truth),
        balanced_accuracy=float(balanced_accuracy_score(truth, prediction)),
        direction_accuracy=float(np.mean(truth == prediction)),
        mcc=float(matthews_corrcoef(truth, prediction)),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        predicted_up_share=float(np.mean(prediction)),
        brier=float(np.mean((score - truth) ** 2)),
    )


def build_constituent_worker(
    *,
    asset_count: int,
    window: int = WINDOW,
    feature_count: int = len(CONSTITUENT_FEATURES),
):
    import tensorflow as tf

    if min(asset_count, window, feature_count) < 1:
        raise ValueError("Constituent worker dimensions must be positive")
    sequence = tf.keras.layers.Input(
        shape=(asset_count, window, feature_count), name="constituent_sequence"
    )
    active_mask = tf.keras.layers.Input(shape=(asset_count,), name="active_member_mask")
    encoded = tf.keras.layers.TimeDistributed(
        tf.keras.layers.LSTM(CONSTITUENT_LSTM_UNITS), name="shared_constituent_lstm"
    )(sequence)
    encoded = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(CONSTITUENT_HIDDEN_UNITS, activation="tanh"),
        name="shared_constituent_projection",
    )(encoded)
    raw_scores = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, use_bias=False), name="member_attention_score"
    )(encoded)
    raw_scores = tf.keras.layers.Reshape((asset_count,), name="flat_member_scores")(
        raw_scores
    )

    def _masked_softmax(values):
        scores, mask = values
        masked = tf.where(mask > 0.0, scores, tf.cast(-1e9, scores.dtype))
        return tf.nn.softmax(masked, axis=-1)

    attention = tf.keras.layers.Lambda(
        _masked_softmax,
        output_shape=(asset_count,),
        name="point_in_time_member_attention",
    )([raw_scores, active_mask])

    def _weighted_pool(values):
        member_encoding, weights = values
        return tf.reduce_sum(member_encoding * weights[..., None], axis=1)

    pooled = tf.keras.layers.Lambda(
        _weighted_pool,
        output_shape=(CONSTITUENT_HIDDEN_UNITS,),
        name="bottom_up_pool",
    )([encoded, attention])
    hidden = tf.keras.layers.Dense(
        CONSTITUENT_HIDDEN_UNITS,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="bottom_up_hidden",
    )(pooled)
    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction")(hidden)
    breadth = tf.keras.layers.Dense(1, activation="sigmoid", name="next_breadth")(hidden)
    model = tf.keras.Model(
        inputs={"constituent_sequence": sequence, "active_member_mask": active_mask},
        outputs={"direction": direction, "next_breadth": breadth},
        name="pit_set50_constituent_worker",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={"direction": "binary_crossentropy", "next_breadth": "mse"},
        loss_weights={"direction": 1.0, "next_breadth": BOTTOM_AUXILIARY_WEIGHT},
        metrics={"direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]},
    )
    diagnostics = tf.keras.Model(
        inputs=model.inputs,
        outputs={"direction": direction, "next_breadth": breadth, "attention": attention},
        name="pit_set50_constituent_diagnostics",
    )
    return model, diagnostics


def build_reconciliation_leader(*, top_experts: int):
    import tensorflow as tf

    if top_experts < 2:
        raise ValueError("Reconciliation requires at least two top-down experts")
    top_scores = tf.keras.layers.Input(shape=(top_experts,), name="top_down_scores")
    bottom_logit = tf.keras.layers.Input(shape=(1,), name="bottom_up_logit")
    context = tf.keras.layers.Input(shape=(2,), name="reconciliation_context")
    top_hidden = tf.keras.layers.Dense(
        LEADER_HIDDEN_UNITS,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="top_down_hidden",
    )(top_scores)
    top_logit = tf.keras.layers.Dense(1, name="top_down_logit")(top_hidden)
    combined_context = tf.keras.layers.Concatenate(name="leader_evidence")(
        [top_hidden, bottom_logit, context]
    )
    gate = tf.keras.layers.Dense(1, activation="sigmoid", name="top_down_gate")(
        combined_context
    )
    correction_unit = tf.keras.layers.Dense(
        1, activation="tanh", name="reconciliation_correction_unit"
    )(combined_context)
    correction = tf.keras.layers.Lambda(
        lambda value: value * LEADER_CORRECTION_CAP,
        output_shape=(1,),
        name="bounded_reconciliation_correction",
    )(correction_unit)
    reconciled_logit = tf.keras.layers.Lambda(
        lambda values: values[0] * values[1] + (1.0 - values[0]) * values[2] + values[3],
        output_shape=(1,),
        name="reconciled_logit",
    )([gate, top_logit, bottom_logit, correction])
    probability = tf.keras.layers.Activation("sigmoid", name="probability")(
        reconciled_logit
    )
    model = tf.keras.Model(
        inputs={
            "top_down_scores": top_scores,
            "bottom_up_logit": bottom_logit,
            "reconciliation_context": context,
        },
        outputs=probability,
        name="pit_set50_reconciliation_leader",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    diagnostics = tf.keras.Model(
        inputs=model.inputs,
        outputs={
            "probability": probability,
            "top_down_gate": gate,
            "top_down_logit": top_logit,
            "correction": correction,
        },
        name="pit_set50_reconciliation_diagnostics",
    )
    return model, diagnostics


def build_top_only_stack(*, top_experts: int):
    import tensorflow as tf

    if top_experts < 2:
        raise ValueError("Top-only stack requires at least two experts")
    scores = tf.keras.layers.Input(shape=(top_experts,), name="top_down_scores")
    hidden = tf.keras.layers.Dense(
        LEADER_HIDDEN_UNITS,
        activation="tanh",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="top_only_hidden",
    )(scores)
    probability = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(
        hidden
    )
    model = tf.keras.Model(scores, probability, name="top_only_stack")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model

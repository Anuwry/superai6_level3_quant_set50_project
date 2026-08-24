from __future__ import annotations

import numpy as np
import pytest

from models.tcrc_lstm import (
    TCRCConfig,
    VARIANTS,
    balanced_positive_weight,
    build_tcrc_lstm,
    compute_tcrc_loss,
    turning_point_targets,
)


def test_turning_targets_use_only_current_and_next_observed_moves() -> None:
    close = np.asarray([100.0, 102.0, 101.0, 101.0])
    next_close = np.asarray([102.0, 101.0, 101.0, 103.0])

    labels, valid = turning_point_targets(
        close,
        next_close,
        previous_close=99.0,
    )

    np.testing.assert_array_equal(labels, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_turning_targets_require_boundary_history_for_first_row() -> None:
    with pytest.raises(ValueError, match="previous_close"):
        turning_point_targets(
            np.asarray([100.0, 101.0]),
            np.asarray([101.0, 100.0]),
        )


def test_balanced_positive_weight_fails_closed_without_both_classes() -> None:
    assert balanced_positive_weight(np.asarray([0, 0, 1, 1, 1])) == pytest.approx(
        2.0 / 3.0
    )
    with pytest.raises(ValueError, match="both classes"):
        balanced_positive_weight(np.ones(4))


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_produces_finite_coherent_outputs(variant: str) -> None:
    import torch

    torch.manual_seed(42)
    config = TCRCConfig(window=20, lstm_window=5)
    model = build_tcrc_lstm(
        input_features=7,
        variant=variant,
        return_mean=0.05,
        return_std=1.2,
        config=config,
    )
    batch = torch.randn(8, 20, 7)

    output = model(batch)

    assert output["standardized_return"].shape == (8,)
    assert output["attention_weights"].shape == (8, 5)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert torch.all((output["direction_probability"] >= 0.0))
    assert torch.all((output["direction_probability"] <= 1.0))
    assert torch.all((output["gate"] >= 0.0) & (output["gate"] <= 1.0))
    expected = torch.sigmoid(output["direction_logit"])
    torch.testing.assert_close(output["direction_probability"], expected)
    assert torch.equal(
        output["direction_probability"] > 0.5,
        output["raw_return_percent"] > 0.0,
    )


def test_full_loss_is_finite_and_backpropagates() -> None:
    import torch

    torch.manual_seed(7)
    config = TCRCConfig(window=20, lstm_window=5)
    model = build_tcrc_lstm(
        input_features=4,
        variant="tcrc_full",
        return_mean=0.0,
        return_std=1.0,
        config=config,
    )
    output = model(torch.randn(10, 20, 4))
    targets = {
        "standardized_return": torch.linspace(-1.0, 1.0, 10),
        "direction": torch.tensor([0.0, 1.0] * 5),
        "turn": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0] * 2),
        "turn_valid": torch.tensor(
            [True, True, True, False, True, True, True, True, True, True]
        ),
    }

    terms = compute_tcrc_loss(
        output,
        targets,
        variant="tcrc_full",
        direction_positive_weight=1.0,
        turn_positive_weight=1.0,
        config=config,
    )
    terms["loss"].backward()

    assert set(terms) == {
        "loss",
        "return_loss",
        "direction_loss",
        "anchor_loss",
        "turn_loss",
        "correction_penalty",
    }
    assert all(torch.isfinite(value) for value in terms.values())
    assert any(parameter.grad is not None for parameter in model.parameters())

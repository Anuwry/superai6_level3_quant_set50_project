from __future__ import annotations

import numpy as np
import pytest

from models.track_c_lime_audit import (
    LIME_PROTOCOL_VERSION,
    LimeAuditConfig,
    aggregate_local_shap,
    compare_local_explanations,
    explain_instance_with_lime,
    fit_lime_surrogate,
    generate_grouped_neighborhood,
    lime_repeat_stability,
    run_lime_explainer_smoke,
    select_regime_audit_indices,
)


def test_lime_config_locks_post_selection_audit_contract():
    config = LimeAuditConfig()

    assert config.protocol_version == LIME_PROTOCOL_VERSION
    assert config.n_perturbations == 1024
    assert config.repeat_seeds == (42, 123, 456, 789, 2025)
    assert config.samples_per_regime_fold == 6
    assert config.top_k == 10
    assert config.selection_role == "post_selection_audit_only"


def test_grouped_neighborhood_preserves_complete_feature_trajectories():
    instance = np.arange(12, dtype=np.float32).reshape(3, 4)
    background = np.stack(
        [
            np.full((3, 4), 100.0, dtype=np.float32),
            np.full((3, 4), 200.0, dtype=np.float32),
        ]
    )

    first = generate_grouped_neighborhood(
        instance,
        background,
        n_perturbations=16,
        seed=42,
    )
    second = generate_grouped_neighborhood(
        instance,
        background,
        n_perturbations=16,
        seed=42,
    )

    assert np.array_equal(first.masks, second.masks)
    assert np.array_equal(first.perturbed, second.perturbed)
    assert np.array_equal(first.masks[0], np.ones(4))
    assert np.array_equal(first.perturbed[0], instance)
    assert np.all(first.weights > 0.0)
    assert first.weights[0] == pytest.approx(1.0)

    for row_index in range(1, len(first.masks)):
        mask = first.masks[row_index]
        perturbed = first.perturbed[row_index]
        for feature_index, is_present in enumerate(mask):
            if is_present:
                assert np.array_equal(
                    perturbed[:, feature_index],
                    instance[:, feature_index],
                )
            else:
                candidates = {
                    tuple(path)
                    for path in background[:, :, feature_index]
                }
                assert tuple(perturbed[:, feature_index]) in candidates


def test_grouped_neighborhood_does_not_mutate_inputs():
    instance = np.arange(6, dtype=np.float32).reshape(2, 3)
    background = np.ones((2, 2, 3), dtype=np.float32)
    original_instance = instance.copy()
    original_background = background.copy()

    generate_grouped_neighborhood(
        instance,
        background,
        n_perturbations=8,
        seed=42,
    )

    assert np.array_equal(instance, original_instance)
    assert np.array_equal(background, original_background)


@pytest.mark.parametrize(
    ("instance_shape", "background_shape"),
    [
        ((3,), (2, 3, 1)),
        ((2, 3), (2, 4, 3)),
        ((2, 3), (2, 2, 4)),
    ],
)
def test_grouped_neighborhood_rejects_invalid_shapes(
    instance_shape: tuple[int, ...],
    background_shape: tuple[int, ...],
):
    instance = np.zeros(instance_shape, dtype=np.float32)
    background = np.zeros(background_shape, dtype=np.float32)

    with pytest.raises(ValueError):
        generate_grouped_neighborhood(
            instance,
            background,
            n_perturbations=8,
            seed=42,
        )


def test_grouped_neighborhood_rejects_nonfinite_values():
    instance = np.zeros((2, 3), dtype=np.float32)
    background = np.zeros((2, 2, 3), dtype=np.float32)
    background[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        generate_grouped_neighborhood(
            instance,
            background,
            n_perturbations=8,
            seed=42,
        )


def test_lime_surrogate_recovers_local_linear_behavior():
    rng = np.random.default_rng(7)
    masks = rng.integers(0, 2, size=(400, 4)).astype(float)
    masks[0] = 1.0
    true_coefficients = np.array([2.0, -1.5, 0.0, 0.5])
    outputs = 3.0 + masks @ true_coefficients
    weights = np.exp(-np.mean(1.0 - masks, axis=1))

    result = fit_lime_surrogate(
        masks,
        outputs,
        weights,
        ridge_alpha=1e-8,
    )

    assert result.fidelity_r2 > 0.999999
    assert result.intercept == pytest.approx(3.0, abs=1e-5)
    assert result.coefficients == pytest.approx(
        true_coefficients,
        abs=1e-5,
    )


def test_grouped_lime_end_to_end_explains_predicted_change():
    instance = np.ones((2, 3), dtype=np.float32)
    background = np.zeros((4, 2, 3), dtype=np.float32)

    def predict_change(sequences: np.ndarray) -> np.ndarray:
        return (
            2.0 * sequences[:, :, 0].sum(axis=1)
            - sequences[:, :, 1].sum(axis=1)
            + 0.5 * sequences[:, :, 2].sum(axis=1)
        )

    result = explain_instance_with_lime(
        predict_change,
        instance,
        background,
        n_perturbations=2048,
        seed=42,
        ridge_alpha=1e-6,
    )

    assert result.black_box_prediction == pytest.approx(3.0)
    assert result.surrogate.fidelity_r2 > 0.999999
    assert result.surrogate.local_prediction == pytest.approx(3.0, abs=1e-5)
    assert result.surrogate.coefficients == pytest.approx(
        [4.0, -2.0, 1.0],
        abs=1e-5,
    )
    assert result.runtime_seconds >= 0.0
    assert result.inference_seconds >= 0.0
    assert result.inference_seconds <= result.runtime_seconds


def test_lime_repeat_stability_uses_all_seed_pairs():
    coefficients = np.array(
        [
            [4.0, -3.0, 2.0, 0.1],
            [3.9, -3.1, 2.1, 0.2],
            [4.1, -2.9, 1.9, 0.0],
        ]
    )

    result = lime_repeat_stability(coefficients, top_k=2)

    assert result["n_repeats"] == 3
    assert result["n_pairs"] == 3
    assert result["pairwise_top_k_jaccard_median"] == pytest.approx(1.0)
    assert result["pairwise_spearman_abs_median"] == pytest.approx(1.0)
    assert result["mean_feature_coefficient_std"] > 0.0
    assert result["median_feature_relative_std"] >= 0.0
    assert result["max_feature_coefficient_std"] > 0.0


def test_aggregate_local_shap_sums_lags_and_preserves_output_additivity():
    values = np.array(
        [
            [[1.0, -1.0], [2.0, 0.5]],
            [[-2.0, 1.0], [0.5, 3.0]],
        ]
    )

    aggregated = aggregate_local_shap(values)

    assert aggregated.shape == (2, 2)
    np.testing.assert_allclose(
        aggregated,
        [[3.0, -0.5], [-1.5, 4.0]],
    )
    np.testing.assert_allclose(
        aggregated.sum(axis=1),
        values.sum(axis=(1, 2)),
    )


def test_compare_local_explanations_reports_rank_overlap_and_sign():
    shap_values = np.array([4.0, -3.0, 2.0, 0.1])
    lime_values = np.array([3.0, -2.0, 1.0, -0.2])

    metrics = compare_local_explanations(
        shap_values,
        lime_values,
        top_k=2,
    )

    assert metrics["spearman_abs"] == pytest.approx(1.0)
    assert metrics["top_k_jaccard"] == pytest.approx(1.0)
    assert metrics["sign_agreement_nonzero"] == pytest.approx(0.75)


def test_select_regime_indices_is_outcome_independent_and_evenly_spaced():
    regimes = np.array(
        ["Bull"] * 8 + ["Sideway"] * 5 + ["Bear"] * 7,
        dtype=object,
    )
    first = select_regime_audit_indices(
        regimes,
        samples_per_regime=3,
    )
    second = select_regime_audit_indices(
        regimes,
        samples_per_regime=3,
    )

    assert first == second
    assert set(first) == {"Bull", "Sideway", "Bear"}
    assert all(len(indices) == 3 for indices in first.values())
    assert first["Bull"][0] == 0
    assert first["Bull"][-1] == 7


def test_select_regime_indices_fails_closed_when_a_regime_is_too_small():
    regimes = np.array(["Bull"] * 6 + ["Sideway"] * 5 + ["Bear"] * 6)

    with pytest.raises(ValueError, match="Sideway"):
        select_regime_audit_indices(
            regimes,
            samples_per_regime=6,
        )


def test_five_model_lime_smoke_is_finite_and_does_not_generate_ranking(
    tmp_path,
):
    payload = run_lime_explainer_smoke(
        output_path=tmp_path / "lime_smoke.json",
        window=3,
        features=6,
        n_perturbations=32,
    )

    assert payload["protocol_version"] == LIME_PROTOCOL_VERSION
    assert payload["ranking_generated"] is False
    assert payload["outer_explanations_generated"] is False
    assert payload["all_models_passed"] is True
    assert len(payload["results"]) == 5
    assert all(row["finite"] for row in payload["results"])
    assert all(row["repeat_exact"] for row in payload["results"])

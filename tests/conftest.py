import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PRIVATE_ARTIFACT_REQUIREMENTS: dict[str, tuple[Path, ...]] = {
    "tests/test_full_ta_feature_pool.py::test_generated_full_ta_folds_are_ready": (
        Path("data-folds-full-ta"),
    ),
    "tests/test_full_ta_feature_pool.py::test_generated_full_ta_nn_folds_are_scaled_train_only": (
        Path("data-folds-full-ta-nn"),
    ),
    "tests/test_integrated_multimodal.py::test_frozen_manifest_matches_current_inputs": (
        Path("outputs/track_b/forward_news/daily_news_features_2019_2025.csv"),
    ),
    "tests/test_integrated_multimodal.py::test_actual_fold_preparation_has_common_dates_and_122_plus_8_features": (
        Path("data-folds-full-ta-vmd-point-in-time-v2"),
    ),
    "tests/test_pit_cmm_extension.py::test_extension_freeze_hashes_match_current_inputs": (
        Path("outputs/track_b/forward_news/daily_news_features_2019_2025.csv"),
    ),
    "tests/test_pit_dern_extension.py::test_freeze_hashes_match_current_inputs": (
        Path("outputs/track_b/forward_news/daily_news_features_2019_2025.csv"),
    ),
    "tests/test_pit_fcg_runner.py::test_load_development_frame_is_locked_to_2019_2021_and_122_numeric_features": (
        Path("data-folds-full-ta-vmd-point-in-time-v2"),
    ),
    "tests/test_tcrc_lstm_runner.py::test_registered_freeze_precedes_tcrc_results": (
        Path("outputs/track_b/forward_news/daily_news_features_2019_2025.csv"),
    ),
    "tests/test_time_based_validation.py::test_full_ta_tuning_folds_have_strict_temporal_order": (
        Path("data-folds-full-ta-validation-point-in-time-v2"),
    ),
    "tests/test_time_based_validation.py::test_full_ta_nn_scaler_is_fit_on_inner_train_only": (
        Path("data-folds-full-ta-validation-point-in-time-v2-nn"),
    ),
    "tests/test_track_b_forward_news.py::test_official_membership_pdf_parser_finds_ranked_set50_constituents": (
        Path("data-raw/track_b/SET50_membership_2024_2025/SET50_100_H1_2024.pdf"),
    ),
    "tests/test_track_c_lime_integrity.py::test_completed_lime_artifacts_pass_integrity_audit": (
        Path("outputs/track_c/dual_xai_lime_v1/local_explanations.csv"),
    ),
    "tests/test_track_c_outer_audit.py::test_completed_outer_artifacts_pass_integrity_audit": (
        Path(
            "outputs/track_c/outer_v2/cells/lstm/fold_1/seed_42/"
            "predictions_Global-All.csv"
        ),
    ),
    "tests/test_vmd_feature_pool.py::test_generated_vmd_folds_are_finite_and_scaled_train_only": (
        Path("data-folds-full-ta-vmd"),
    ),
}


def missing_private_artifacts(nodeid: str, project_root: Path) -> tuple[Path, ...]:
    requirements = PRIVATE_ARTIFACT_REQUIREMENTS.get(nodeid, ())
    return tuple(path for path in requirements if not (project_root / path).exists())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        missing = missing_private_artifacts(item.nodeid, PROJECT_ROOT)
        if not missing:
            continue
        item.add_marker(pytest.mark.private_artifact)
        joined = ", ".join(path.as_posix() for path in missing)
        item.add_marker(
            pytest.mark.skip(
                reason=f"requires non-redistributable research artifact(s): {joined}"
            )
        )

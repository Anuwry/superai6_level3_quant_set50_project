import numpy as np
import pandas as pd

from models.baseline_common import FoldData, FoldSpec
from models.persistence_baseline import predict_persistence_fold


def test_persistence_prediction_equals_current_close(tmp_path):
    spec = FoldSpec(
        fold="fold_1",
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        train_start_year=2020,
        train_end_year=2021,
        test_year=2022,
    )
    train = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01"]),
            "Close_D": [100.0],
            "Target_Next_Close": [101.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
            "Close_D": [110.0, 112.0],
            "Target_Next_Close": [112.0, 111.0],
        }
    )
    fold = FoldData(spec=spec, train=train, test=test, feature_columns=["Close_D"])

    prediction = predict_persistence_fold(fold)

    np.testing.assert_array_equal(prediction, test["Close_D"].to_numpy())

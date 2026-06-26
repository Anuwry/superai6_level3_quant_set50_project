from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import CLOSE_COLUMN, FoldData, print_metrics, run_model_on_folds

MODEL_NAME = "chronos_t5_tiny_zero_shot_greedy"
MODEL_ID = "amazon/chronos-t5-tiny"
CONFIG = {
    "experiment": "naive_baseline",
    "model": "Chronos T5 Tiny",
    "model_id": MODEL_ID,
    "hyperparameter_tuning": False,
    "fine_tuning": False,
    "zero_shot": True,
    "sampling": False,
    "search": "greedy",
    "prediction_length": 1,
}


def chronos_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline():
    import torch
    from chronos import ChronosPipeline

    device = chronos_device()
    return ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )


def predict_next_value_greedy(pipeline, context_values: np.ndarray) -> float:
    import torch
    from transformers import GenerationConfig

    context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
    token_ids, attention_mask, scale = pipeline.tokenizer.context_input_transform(context_tensor)
    token_ids = token_ids.to(pipeline.model.device)
    attention_mask = attention_mask.to(pipeline.model.device)
    generation = pipeline.model.model.generate(
        input_ids=token_ids,
        attention_mask=attention_mask,
        generation_config=GenerationConfig(
            min_new_tokens=1,
            max_new_tokens=1,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=pipeline.model.config.eos_token_id,
            pad_token_id=pipeline.model.config.pad_token_id,
        ),
    )
    if pipeline.model.config.model_type == "seq2seq":
        generation = generation[..., 1:]
    else:
        generation = generation[..., -1:]
    samples = generation.reshape(1, 1, 1).to(scale.device)
    prediction = pipeline.tokenizer.output_transform(samples, scale)
    return float(prediction.reshape(-1)[0].item())


def predict_fold(fold: FoldData) -> np.ndarray:
    pipeline = load_pipeline()
    train_close = fold.train[CLOSE_COLUMN].to_numpy(dtype=float)
    test_close = fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    predictions = []
    for index in range(len(test_close)):
        context = np.concatenate([train_close, test_close[: index + 1]])
        predictions.append(predict_next_value_greedy(pipeline, context))
    return np.asarray(predictions, dtype=float)


def main():
    metrics = run_model_on_folds(MODEL_NAME, predict_fold, CONFIG, ["numpy", "pandas", "torch", "transformers", "chronos-forecasting"])
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()

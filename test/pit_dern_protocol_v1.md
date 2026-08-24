# PIT-DERN Exploratory Extension Protocol v1

Protocol ID: `pit-dern-exploratory-v1`  
Evidence status: `post_freeze_exploratory_architecture_extension`  
Freeze date: 2026-08-04 (Asia/Bangkok)  
Result access at this protocol freeze: **no PIT-DERN result existed**

## Objective

Evaluate whether a Point-in-Time Dual-Evidence Retrieval Network (PIT-DERN)
can improve next-trading-day SET50 direction prediction over the five frozen
neural benchmarks without changing their registered results.

PIT-DERN is a supervised hybrid neural/retrieval classifier. A lightweight
dual-axis mixer learns a query embedding and two auxiliary outputs: next-day
direction probability and scaled next-close delta. At inference, each query
retrieves evidence independently from historical up and down memory banks.
The class-balanced evidence is combined with the neural prediction through a
fixed transferability gate. Retrieval memory contains training observations
only; a label is eligible only after its label date precedes the query date.

## Fixed evaluation contract

- Primary metric: balanced accuracy (BAcc).
- Primary arm: `Regime-SHAP-Numeric-News`.
- Primary comparator: frozen `LSTM-CNN-Attention`, the current best final-arm
  BAcc among the registered five models.
- Secondary metrics: direction accuracy, MCC, RMSE, and MAE.
- Temporal folds: held-out years 2022, 2023, 2024, and 2025.
- Seeds: 42, 123, 456, 789, and 2025.
- Sequence window: 5 trading days.
- Decision threshold: 0.5, fixed before execution.
- Existing five-model rows are read from the frozen integrated result table;
  they are not rerun or changed.

## Fixed architecture and training

- Input projection dimension: 24.
- Dual-axis mixer blocks: 2.
- Mixer expansion: 2x.
- Embedding dimension: 16 with L2 normalization.
- Dropout: 0.10.
- Epochs: 20.
- Batch size: 32.
- Optimizer: Adam, learning rate 0.001.
- Direction loss: binary cross-entropy, weight 1.0.
- Scaled next-close-delta loss: Huber, weight 0.25.
- Supervised contrastive embedding loss: temperature 0.10, weight 0.10.
- Shuffle: false.
- No Optuna, early stopping, test-threshold tuning, or post-result feature
  selection.

## Fixed retrieval mechanism

- Similarity: cosine similarity between L2-normalized embeddings.
- Standard retrieval ablation: top 10 observations across the full memory.
- Dual evidence: top 5 from the up bank and top 5 from the down bank.
- Retrieval temperature: 0.20.
- Minimum similarity anchor for the gate: 0.25.
- Transferability gate: fixed deterministic combination of best-bank
  similarity quality and dual-evidence probability margin.
- Retrieval delta: probability-weighted class-conditional softmax average of
  known training deltas.
- Final direction sign is determined by the final probability; magnitude is
  determined by the absolute blended scaled delta.

## Regime and SHAP treatment

Global arms use the complete registered numerical pool, with the news block
added only to news arms. Regime-SHAP arms use a single shared model rather than
three separately fitted experts:

- the ordered union of the three frozen regime-specific SHAP feature sets;
- features selected for the row's point-in-time regime receive weight 1.0;
- other union features retain a fixed residual weight of 0.25;
- the frozen causal regime probabilities for bull, sideway, and bear are
  appended as context features;
- news features remain unmasked.

The final model therefore uses regime information without fragmenting the
training sample into regime-specific neural fits.

## Registered ablations

1. `encoder_only`: dual-axis neural encoder without retrieval.
2. `standard_retrieval`: ordinary top-10 retrieval over the unpartitioned
   memory.
3. `dual_evidence`: balanced top-5 up and top-5 down evidence, without the
   neural fallback.
4. `pit_dern`: fixed transferability blend of encoder and dual evidence.
5. `shuffled_retrieval_control`: memory labels/deltas permuted relative to
   embeddings using the registered seed; model fitting is unchanged.

## Promotion gates

PIT-DERN is promoted only if all gates pass:

1. Mean final-arm BAcc is at least 1.0 percentage point above the frozen
   LSTM-CNN-Attention final-arm BAcc.
2. Final-arm BAcc delta versus LSTM-CNN-Attention is positive in at least
   three of four temporal folds.
3. PIT-DERN final-arm BAcc exceeds `encoder_only` final-arm BAcc.
4. PIT-DERN final-arm BAcc exceeds the label-shuffled retrieval control.
5. Trainable parameters are no more than 15% above the frozen
   LSTM-CNN-Attention parameter count for the matched arm input.
6. All required predictions are complete and finite.

Failure to pass does not invalidate the run. It means PIT-DERN remains a
reported exploratory negative or mixed architecture result and cannot replace
the frozen primary model.

## Runtime and cost contract

- Record fit, embedding, retrieval, inference, and isolated-cell wall times.
- Record trainable parameter counts and retrieval memory sizes.
- Native Windows TensorFlow CPU execution is acceptable and must be logged.
- Incremental API cost is fixed at USD 0; all news features are already frozen.

## Required outputs

- seed/fold/arm metrics and prediction files;
- retrieval evidence diagnostics, including retrieved training dates;
- seed-averaged fold metrics;
- ablation and shuffled-control tables;
- six-model all-arm and final-arm comparisons;
- parameter/runtime comparisons;
- promotion decision and integrity audit;
- execution log suitable for paper reporting.

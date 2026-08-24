# SET50 Next-Day Direction Prediction

ช่วงข้อมูลตลาด: 22/01/2010–22/12/2025 ครอบคลุมข้อมูลรายวัน รายสัปดาห์
และรายเดือน โดยทำนายราคาปิดวันถัดไปและแปลงผลเป็นทิศทาง Up/Down

โมเดลที่ลงทะเบียนใน pipeline และ paper มีเพียง 5 โมเดล:

- LSTM
- CNN
- LSTM-CNN
- LSTM-Attention
- LSTM-CNN-Attention

Track A เปรียบเทียบ Full TA กับ Full TA + causal rolling VMD ภายใต้
expanding-window evaluation ปี 2022–2025 และเลือก sliding window จาก
`1, 3, 5, 10, 20` ด้วยข้อมูลก่อนปี 2022 เท่านั้น

ทุก split ใช้ point-in-time label purge:

```text
retain supervised training row only if Label_Date < first evaluation Date
```

feature ของ boundary row ที่ถูก purge จะใช้เป็น sequence context เท่านั้น
ไม่ถูกส่งเข้า `model.fit` และ scaler fit เฉพาะ supervised train partition

Track B เพิ่ม sentiment/news แบบ paired ablation บน 5 โมเดลเดิม ส่วน Track C
ทำ Bull/Sideway/Bear routing และ progressive SHAP refinement โดย SHAP
อธิบาย predicted next-day change ไม่ใช่ระดับราคาปิดดิบ

Post-hoc integrated extension รวม Track B และ Track C จริงบน common cohort
เริ่มปี 2019 ด้วย 2 x 2 arms: Global/Regime-SHAP x Numeric/+News ครบ 5
โมเดล 4 folds 5 seeds (100 cells, 800 fits) Downstream news features มาจาก
frozen expanding Local NLP; LLM single/debate เป็น intrinsic benchmark แยก
ต่างหากและไม่ได้ถูกอ้างเป็น feature source ค่า API เพิ่มของ integrated run =
USD 0 ผล BAcc ดีขึ้นภายใน regime pipeline เพียง 2/5 โมเดลและไม่มี contrast
ใดผ่าน Holm correction

ผลทดลองบันทึก runtime, seed, fold, window, Direction Accuracy, Balanced
Accuracy, MCC, coverage, RMSE, MAE และ metadata ที่ใช้ทำซ้ำการทดลอง

คำสั่งทำซ้ำ pipeline แบบแยก process เพื่อลด TensorFlow memory growth:

```powershell
py -3.12 -m models.run_five_model_pipeline all --force
```

คำสั่ง integrated runner แบบ resumable (ใช้ Python environment ที่มี
TensorFlow 2.21.0):

```powershell
python -m models.integrated_multimodal_runner run
python -m models.integrated_multimodal_runner aggregate
```

หลักฐานควบคุมอยู่ที่ `test/integrated_multimodal_protocol_v1.md`,
`test/integrated_multimodal_execution_log_v1.md` และ
`outputs/integrated_multimodal_posthoc_v1/integrity_audit.json`

Market-data governance และ SET100 point-in-time preparation:

```powershell
$env:PYTHONPATH=(Get-Location).Path
D:\conda_envs\my_env\python.exe scripts\run_market_data_governance.py
```

ผล governance แบบ machine-readable อยู่ที่
`outputs/market_data_governance_v1/` และ execution/paper log อยู่ที่
`test/market_data_governance_v1.md` ข้อมูล SET100 แบบ row-level และ folds เป็น
restricted artifacts จึงถูกเก็บใน `set100_data/` และไม่ถูก track ใน Git

SET100 same-exchange robustness benchmark เสร็จแล้วภายใต้ protocol ที่ freeze
ก่อนเห็นผล: 5 โมเดล x 4 outer folds (2022–2025) x 5 seeds = 100 fits โดยใช้
Full TA + causal VMD และ windows เดิมจาก SET50 ผล SET100 mean BAcc อยู่ระหว่าง
0.5003–0.5189 และต่ำกว่า SET50 ทั้ง 5 โมเดลเฉลี่ย 0.95–2.17 percentage points;
ไม่มี paired cross-index comparison ที่ผ่าน Holm correction ที่ 0.05 งานนี้จึง
เป็น same-exchange transfer/reliability audit ไม่ใช่ external-market replication
หรือหลักฐานว่า accuracy ดีขึ้น รายละเอียดอยู่ที่
`test/set100_same_exchange_robustness_execution_log_v1.md` และผลรวมแบบ
machine-readable อยู่ที่ `outputs/set100_same_exchange_robustness_v1/`

```powershell
py -3.12 -m models.set100_robustness_runner prepare
py -3.12 -m models.set100_robustness_runner run
```

## Strong-Q2 hardening update (2026-08-03)

The frozen Track B LLM compute-matched control is complete on 1,333 labelled
article-ticker pairs (738 unique articles). The Bull/Bear/Leader system reached
76.594% intrinsic sentiment Accuracy versus 70.668% for an equal-call
three-pass self-consistency control. The paired difference was +5.926
percentage points with an article-cluster 95% interval [3.491, 8.487] and
Holm-adjusted p=0.000040. A four-pass near-cost control used 93.0% of the Leader
cost and produced the same conclusion (+6.002 pp, 95% interval [3.613, 8.477],
Holm-adjusted p=0.000040).

This intrinsic LLM result is not the source of downstream forecasting news
features. The integrated forecasting experiment continues to use frozen
expanding Local NLP features, and its Balanced Accuracy conclusions remain
mixed/null.

The existing integrated temporal sensitivity was also audited: five models x
five contrasts x two metrics, 10-day circular moving blocks, and 10,000
replicates. None of the 25 Balanced Accuracy contrasts passed Holm correction;
significant squared-error rows must not be reported as directional evidence.

Public-release packaging is fail-closed. The generated bundle contains code,
tests, protocols, hashes, and non-reconstructive aggregate evidence only. Raw
SET50/SET100 rows, point-in-time fold CSVs, row-level predictions, private LLM
checkpoints, and keys are excluded. Market data are described as publicly
accessible provider data, not as open-licensed data, and row-level provider
observations are not redistributed.

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 scripts/build_public_replication_package.py
```

Key audit documents:

- `test/strong_q2_claims_register_v2.md`
- `test/strong_q2_hardening_execution_log_v1.md`
- `test/track_b_compute_matched_execution_log_v1.md`
- `test/moving_block_bootstrap_audit_v1.md`
- `PUBLIC_REPLICATION_PACKAGE.md`

## Multimodal falsification and reproducibility update (2026-08-04)

The pre-frozen retrospective falsification extension is complete for all five
architectures, four outer years, and five seeds: 100/100 cells and 400 new
fits. It adds News-Only, joint shuffled-news, five-trading-row lagged-news, and
eight-random-feature controls while reusing the exact persisted Market-Only
and Observed-News reference predictions. No one of 30 Balanced Accuracy rows
survived Holm adjustment under either exact four-fold inference or the 10-day,
10,000-replicate moving-block sensitivity. Incremental API cost was USD 0.

This is main-text reliability evidence: observed news did not establish a
stable incremental directional benefit. It does not negate the separate
intrinsic LLM benchmark and does not turn shuffled news into a proposed model.

Manuscript tables and the public package are generated and independently
audited with:

```powershell
py -3.12 scripts/build_manuscript_artifacts.py
py -3.12 scripts/build_public_replication_package.py
py -3.12 scripts/audit_public_replication_package.py
```

Authoritative additions:

- `test/reliability_hardening_execution_log_v2.md`
- `test/strong_q2_claims_register_v3.md`
- `outputs/multimodal_falsification_v1/`
- `outputs/manuscript_tables_v1/`

# Track C: Leakage-Free Market Regime Labeling

สถานะ: **เก็บเป็น HMM v1 baseline; ไม่ใช้เป็น primary router**  
วันที่รัน: 31 กรกฎาคม 2026 (Asia/Bangkok)  
ขอบเขตของ log นี้: การระบุ Bull / Sideway / Bear ด้วย Hidden Markov Model (HMM) เท่านั้น  
สิ่งที่ยังไม่รวม: Progressive SHAP, regime-specific feature sets และการฝึก prediction experts แยกตาม regime

> การตรวจ semantic validity หลังรันพบว่า state ที่ map เป็น Sideway มี volatility
> สูงที่สุดและไม่มี hard Sideway label ใน outer-test ปี 2022–2023 จึงไม่ตรงกับนิยาม
> range-bound market ที่ต้องการ ผลชุดนี้ยังคงเก็บครบเพื่อเป็น baseline/ablation แต่ถูกแทนที่
> เป็น primary router ด้วย daily multi-timescale semantic regime v2 ซึ่งบันทึกไว้ที่
> `test/track_c_daily_regime_v2.md`

## 1. วัตถุประสงค์

สร้าง market-regime feature/routing variable ที่ใช้ข้อมูลได้จริง ณ เวลา \(t\) เพื่อเลือกชุด
features หรือโมเดลสำหรับทำนายทิศทาง SET50 ในวันถัดไป โดยต้องไม่ใช้ target และไม่เห็นข้อมูล
อนาคตของ outer-test fold

แนวทางอ้างอิงหลัก:

1. Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time
   Series and the Business Cycle*: พื้นฐาน Markov-switching regime model  
   <https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html>
2. Wang, Lin, and Mikhelson (2020), *Regime-Switching Factor Investing with Hidden
   Markov Models*: การใช้ HMM เพื่อแยกสภาวะตลาด  
   <https://doi.org/10.3390/jrfm13120311>
3. Werge (2021), *Gaussian Hidden Markov Models on Financial Markets*: การประยุกต์
   Gaussian HMM กับข้อมูลตลาดการเงิน  
   <https://arxiv.org/abs/2107.05535>

จำนวน 3 states และ EWMA span 30 ถูกล็อกจากคำถามวิจัยและ literature ก่อนเปิดผล outer
test ไม่ได้เลือกจากปีทดสอบ 2022–2025

## 2. Data contract และ walk-forward folds

แหล่งข้อมูลคือ `data-folds-full-ta-vmd` แต่ regime labeler อ่านเฉพาะ:

- `Date`
- `Close_D`
- `Close_D_lag1`

ไม่ใช้ `target`, label ทิศทาง, VMD component, technical indicator หรือ news sentiment ใน
การ fit และ label regime

| Fold | Training period | Test period | Train rows | Test rows |
|---|---:|---:|---:|---:|
| 1 | 2012-05-03–2021-12-30 | 2022-01-04–2022-12-30 | 2,359 | 241 |
| 2 | 2012-05-03–2022-12-30 | 2023-01-03–2023-12-28 | 2,600 | 243 |
| 3 | 2012-05-03–2023-12-28 | 2024-01-02–2024-12-30 | 2,843 | 244 |
| 4 | 2012-05-03–2024-12-30 | 2025-01-02–2025-12-18 | 3,087 | 234 |

รวม outer-test 962 trading days

## 3. Locked methodology

### 3.1 Causal observations

คำนวณ log return จากราคาที่มีอยู่แล้ว ณ วัน \(t\):

\[
r_t = \log\left(\frac{Close_t}{Close_{t-1}}\right)
\]

จากนั้นสร้าง observation vector สองมิติ:

\[
x_t =
\left[
EWMA_{30}(r_t),
\sqrt{EWMA_{30}(r_t^2)-EWMA_{30}(r_t)^2}
\right]
\]

ใช้ `adjust=False` และ `min_periods=1`; ค่าของแถว \(t\) จึงขึ้นกับข้อมูลถึง \(t\)
เท่านั้น

### 3.2 Model fitting

- Gaussian HMM จำนวน 3 hidden states
- Full covariance matrix
- StandardScaler fit เฉพาะ training observations ของแต่ละ fold
- HMM fit เฉพาะ training period ของ fold นั้น
- 5 deterministic restarts: seeds 42, 123, 456, 789, 2025
- เลือก restart ที่มี training log-likelihood สูงสุด
- Maximum 200 EM iterations, tolerance \(10^{-4}\), minimum covariance \(10^{-5}\)
- ไม่ใช้ outer-test เพื่อเลือก seed, hyperparameter หรือ state mapping

### 3.3 Semantic state mapping

ซ่อน state ID ของ HMM ไม่ให้ผูกกับชื่อ regime โดยตรง แล้ว map จาก emission mean ของ
training EWMA return:

- mean ต่ำสุด = Bear
- meanกลาง = Sideway
- meanสูงสุด = Bull

mapping ถูกสร้างใหม่ภายใน training fold ทุก fold จึงไม่สมมติว่า hidden state ID เดิมมี
ความหมายเดิม

### 3.4 Causal inference และ next-day routing

ไม่ได้ใช้ Viterbi path หรือ smoothed posterior กับ test data เพราะทั้งสองวิธีอาจทำให้ label
ของวันก่อนหน้าเปลี่ยนเมื่อเห็นข้อมูลอนาคต ใช้ forward-filtered posterior เท่านั้น:

\[
P(S_t \mid x_{1:t})
\]

เนื่องจากงานหลักทำนายวันถัดไป router ที่บันทึกใน `routing_regime` มาจาก:

\[
P(S_{t+1}\mid x_{1:t})
=
P(S_t\mid x_{1:t})A
\]

โดย \(A\) คือ transition matrix ที่ fit จาก training fold ทำให้ prediction row ณ วัน \(t\)
ไม่ใช้ราคา ข่าว หรือ target ของวัน \(t+1\)

ไฟล์รายวันเก็บทั้ง filtered probability ปัจจุบัน, next-day probability, hard regime,
confidence และ entropy เพื่อรองรับทั้ง hard และ soft routing ในขั้นต่อไป

## 4. Quality gates

กำหนด gate ล่วงหน้า:

- HMM restart ที่เลือกต้อง converge
- ทุก training regime ต้องมีสัดส่วนอย่างน้อย 5%
- ทุก state ต้องมี self-transition probability อย่างน้อย 0.80
- probability ทุกแถวต้อง finite และรวมได้ 1

| Fold | Selected seed | Train log-likelihood | Minimum train share | Minimum self-transition | Gate | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 789 | -3,592.008 | 14.67% | 0.9717 | Pass | 7.682 |
| 2 | 456 | -3,943.420 | 15.08% | 0.9774 | Pass | 5.237 |
| 3 | 456 | -4,353.650 | 13.19% | 0.9726 | Pass | 6.496 |
| 4 | 123 | -4,786.198 | 12.60% | 0.9681 | Pass | 6.460 |

ผล: **ผ่านครบ 4/4 folds**  
Runtime รวมของ experiment: **26.571 seconds**

## 5. Outer-test regime distribution

ตารางนี้เป็นผล diagnostic ของ locked, causal regime labeler ไม่ใช่ผล accuracy ของ
direction-prediction model

| Test year | Bull | Sideway | Bear | Total |
|---:|---:|---:|---:|---:|
| 2022 | 186 (77.18%) | 0 (0.00%) | 55 (22.82%) | 241 |
| 2023 | 101 (41.56%) | 0 (0.00%) | 142 (58.44%) | 243 |
| 2024 | 172 (70.49%) | 15 (6.15%) | 57 (23.36%) | 244 |
| 2025 | 19 (8.12%) | 93 (39.74%) | 122 (52.14%) | 234 |

Mean routing confidence ของ regime ที่ถูกเลือกใน test subsets อยู่ประมาณ 0.875–0.965

ข้อสังเกตสำคัญ: ไม่มี hard Sideway routing ใน test ปี 2022 และ 2023 แต่ Sideway ไม่ได้
collapse ระหว่าง training เพราะมีสัดส่วน 12.60–15.08% ในทุก training fold ภายใต้
next-day hard router ดังนั้น paper ต้องรายงานศูนย์นี้ตามจริง และห้ามเติมผล
regime-specific metric ที่ไม่มี observation

## 6. Learned state characteristics

ช่วงค่าจาก state profiles ทั้ง 4 folds:

| Regime | Training emission mean: EWMA return | Mean EWMA volatility | Expected duration |
|---|---:|---:|---:|
| Bull | +0.000580 ถึง +0.000852 | 0.00598–0.00638 | 55.7–71.5 วัน |
| Sideway | -0.000288 ถึง -0.000172 | 0.01865–0.01938 | 39.2–44.2 วัน |
| Bear | -0.000729 ถึง -0.000529 | 0.00928–0.01015 | 31.4–48.7 วัน |

state กลางมี expected return ใกล้ศูนย์ตามเกณฑ์ mapping แต่ volatility สูงที่สุดอย่าง
สม่ำเสมอ จึงควรเรียกเชิงตีความว่า **Sideway/Transition regime** ไม่ควรตีความว่าเป็น
ตลาดนิ่งความผันผวนต่ำโดยอัตโนมัติ

## 7. Leakage and causality audit

ดำเนินการตรวจ prefix-invariance บนข้อมูลจริงทุก fold โดย:

1. คำนวณ observations และ filtered probabilities จาก train + full test
2. ตัดข้อมูลที่ประมาณกึ่งกลาง test แล้วคำนวณใหม่โดยใช้ HMM parameters เดิม
3. เปรียบเทียบทุกค่าบน prefix เดียวกัน

| Fold | Prefix rows | Max observation difference | Max filtered-probability difference |
|---|---:|---:|---:|
| 1 | 2,479 | 0.0 | 0.0 |
| 2 | 2,721 | 0.0 | 0.0 |
| 3 | 2,965 | 0.0 | 0.0 |
| 4 | 3,204 | 0.0 | 0.0 |

Maximum probability row-sum numerical error อยู่ที่ \(8.88\times10^{-16}\)

มี unit test เพิ่มเติมที่เปลี่ยน target ทุกแถวแล้วตรวจว่า regime labels ไม่เปลี่ยน เพื่อ
ยืนยันว่า target ไม่เข้าสู่ pipeline

## 8. Restart stability

HMM มี local optimum จึงรัน 5 seeds ต่อ fold และเก็บผลครบทั้งหมด:

- 20/20 restarts converge
- seed 42 ตก local optimum ที่แย่ในทุก fold โดย log-likelihood ต่ำกว่าจุดดีที่สุดประมาณ
  992–1,101 และ Adjusted Rand Index (ARI) เทียบ selected labels เพียง 0.180–0.265
- เมื่อพิจารณา near-optimal restarts ที่ log-likelihood gap ไม่เกิน 10:
  - Fold 1: minimum ARI = 1.000
  - Fold 2: minimum ARI = 0.523
  - Fold 3: minimum ARI = 0.999
  - Fold 4: minimum ARI = 0.790

ดังนั้น multiple restarts มีความจำเป็นและช่วยหลีกเลี่ยง local optimum ที่ชัดเจน แต่ Fold 2
ยังมี regime-assignment uncertainty ระหว่าง near-optimal solutions จึงไม่ควรกล่าวใน paper
ว่า state assignments เสถียรสมบูรณ์ การเลือก highest training likelihood ถูกล็อกและใช้
แบบเดียวกันทุก fold

## 9. Verification

- Track C tests: 7 passed
- Integrated Track A–C regression suite: 75 passed
- Statement coverage ของ `models/track_c_regime_labeling.py`: 89%
- Black: passed
- Ruff: passed
- Output integrity audit: 13 CSV files, 5 JSON files และ daily-label rows 11,851;
  ไม่พบ unexpected non-finite value หรือ probability row ที่รวมไม่เท่ากับ 1
- Test cases ครอบคลุม causal observation, forward filter, state mapping, target
  independence, next-day routing, artifact generation, quality gate และ restart ARI

## 10. Reproducibility and artifacts

คำสั่งติดตั้ง:

```powershell
py -3.12 -m pip install -r requirements-track-c.txt
```

คำสั่งรัน:

```powershell
py -3.12 -m models.track_c_regime_labeling
```

ไฟล์สำคัญ:

- Source: `models/track_c_regime_labeling.py`
- Tests: `tests/test_track_c_regime_labeling.py`
- Environment: `requirements-track-c.txt`
- Fold diagnostics: `outputs/track_c/regime_labeling/fold_summary.csv`
- Daily distributions: `outputs/track_c/regime_labeling/regime_distribution.csv`
- Learned state profiles: `outputs/track_c/regime_labeling/state_profiles.csv`
- Transition matrices: `outputs/track_c/regime_labeling/transition_matrices.csv`
- All restart results: `outputs/track_c/regime_labeling/restart_diagnostics.csv`
- Parameters, versions, source hashes และ protocol:
  `outputs/track_c/regime_labeling/run_metadata.json`
- Daily train/test probabilities และ labels:
  `outputs/track_c/regime_labeling/fold_1` ถึง `fold_4`

`run_metadata.json` บันทึก Python/package versions, SHA-256 ของ input CSV ทุกไฟล์,
locked protocol, source citations, runtime และ output locations

## 11. ข้อกำหนดสำหรับ Progressive SHAP ขั้นถัดไป

ผลในโฟลเดอร์นี้เหมาะสำหรับ outer-fold diagnostics และ final outer-fold training/test
routing แต่การเลือก feature ภายใน outer-training ต้องทำแบบ nested:

1. แบ่ง inner-training/inner-validation ตามเวลา
2. fit HMM ใหม่เฉพาะ inner-training prefix
3. ใช้ causal forward filter สร้าง regime ให้ inner-validation
4. ทำ Progressive SHAP และเลือก features แยก Bull/Sideway/Bear เฉพาะข้อมูลที่อนุญาต
5. ล็อก feature set แล้วประเมิน outer-test เพียงครั้งเดียว

ห้ามใช้ HMM ที่ fit จาก full outer-training เพื่อย้อน label inner-validation แล้วใช้
inner-validation นั้นเลือก feature เพราะจะทำให้ feature-selection protocol เห็น
distribution parameters จากอนาคตของ inner split

เนื่องจากบาง split อาจมี Sideway sample น้อย ขั้นถัดไปต้องกำหนด minimum sample count และ
global-model fallback ไว้ล่วงหน้า ส่วน soft routing จาก posterior probability สามารถเก็บเป็น
secondary experiment ได้ แต่ hard next-day routing ควรเป็น primary experiment เพื่อให้
ablation ตีความง่าย

## 12. Paper-ready conclusion

HMM v1 เป็น leakage-free three-state statistical baseline ที่ผ่าน convergence, regime-size
และ persistence gates แต่ไม่ผ่าน semantic face-validity สำหรับคำว่า Sideway เพราะ state
ดังกล่าวทำหน้าที่คล้าย high-volatility transition state มากกว่า range-bound market จึงไม่ใช้
ผลนี้เป็น router ของ Progressive SHAP การตัดสินใจ downstream ให้ใช้ daily multi-timescale
semantic regime v2 และรายงาน HMM v1 เป็น ablation เพื่อแสดงเหตุผลของ protocol correction
อย่างโปร่งใส

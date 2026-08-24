# Track C: Daily Multi-Timescale Semantic Regime v2

สถานะ: **Selected regime protocol; พร้อมเป็น input ให้ nested Progressive SHAP**  
วันที่รัน: 31 กรกฎาคม 2026 (Asia/Bangkok)  
ขอบเขต: daily Bull / Sideway / Bear routing ณ วัน \(t\) สำหรับทำนาย direction วัน
\(t+1\)

## 1. เหตุผลของ protocol correction

HMM v1 ใช้ EWMA return และ volatility แล้ว map hidden state ที่มี mean return ตรงกลางเป็น
Sideway ผล diagnostic พบว่า state นี้มี volatility สูงที่สุดและไม่มี hard Sideway label ใน
ปี 2022–2023 แม้ปี 2022 SET50 เปลี่ยนทั้งปีเพียงประมาณ +0.58% การตรวจ soft posterior
เพิ่มเติมพบว่า maximum Sideway probability มีเพียง 1.02% ในปี 2022 และ 1.30% ในปี 2023
จึงไม่ใช่ปัญหา hard argmax แต่เป็น construct mismatch: state ที่ได้คือ
high-volatility/transition ไม่ใช่ range-bound Sideway

HMM v1 ถูกเก็บครบใน `outputs/track_c/regime_labeling` เพื่อใช้เป็น statistical
baseline/ablation ส่วน v2 ถูกสร้างเพื่อให้ความหมายของ regime ตรงกับการเลือก feature/model
ตามสภาวะตลาดจริง

ข้อจำกัดด้าน research protocol: การพัฒนา v2 เกิดหลังเห็น semantic failure ของ HMM v1 ใน
outer diagnostics ดังนั้นผล v2 ปี 2022–2025 ต้องเรียกว่า **post-hoc robustness evidence**
ไม่ใช่ untouched confirmatory test อย่างไรก็ตาม exact weights และ Sideway deadband ถูก
screen โดยใช้เฉพาะ development 2012–2019 และ validation 2020–2021 ไม่ได้ optimize จาก
direction-prediction accuracy หรือ target ปี 2022–2025

## 2. Prediction and data contract

หนึ่ง observation คือหนึ่ง trading day \(t\)

- Input มีเฉพาะข้อมูลที่ทราบได้เมื่อปิดตลาดวัน \(t\)
- Output คือ current regime \(R_t\), soft membership และ confidence
- `routing_regime = R_t` ใช้เลือก feature set/expert เพื่อทำนาย direction \(y_{t+1}\)
- ไม่ใช้ `Target_Next_Close`, direction target, ข่าวในอนาคต หรือข้อมูลหลังวัน \(t\)
- Fit Sideway threshold ใหม่เฉพาะ training period ของแต่ละ expanding fold
- Outer-test ไม่ถูกใช้ fit threshold

ใช้ข้อมูลจาก `data-folds-full-ta-vmd` เฉพาะ:

- `Date`, `Close_D`, `Close_D_lag1`
- `Return_1D`, `Return_3D`, `Return_5D`, `Return_10D`, `Return_20D`,
  `Return_60D`
- `Volatility_20`, `Volatility_60`
- `ADX_14`

Return, volatility และ ADX ใน feature pool เดิมถูกสร้างด้วย past-only rolling
transform และผ่าน upstream causality tests แล้ว

## 3. Daily multi-timescale semantic score

ทุกวันมี risk-adjusted trend component:

\[
z_{t,h} =
\frac{Return_{t,h}}
{Volatility_{t,v(h)}\sqrt{h}}
\]

โดย:

| Horizon \(h\) | 1 | 3 | 5 | 10 | 20 | 60 |
|---:|---:|---:|---:|---:|---:|---:|
| Weight \(w_h\) | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.25 |
| Volatility reference \(v(h)\) | 20 | 20 | 20 | 20 | 20 | 60 |

จึงยังตอบสนองต่อข้อมูล 1 วัน แต่ให้น้ำหนักกับโครงสร้าง 10–60 วันมากกว่าเพื่อลดการตีความ
daily noise เป็น market regime

Composite trend:

\[
T_t = \sum_h w_h z_{t,h}
\]

Directional strength:

\[
D_t = ADX_{14,t}/100
\]

Raw semantic score และ causal smoothing:

\[
Q_t = T_tD_t
\]

\[
S_t = EWMA_3(Q_t)
\]

ADX อ้างอิงจาก Wilder (1978), *New Concepts in Technical Trading Systems* ส่วนการใช้
directional return across horizons สอดคล้องกับแนวคิด time-series momentum ของ Moskowitz,
Ooi, and Pedersen (2012), DOI: <https://doi.org/10.1016/j.jfineco.2011.11.003>

## 4. Sideway deadband and routing

ในแต่ละ fold fit threshold จาก training scores เท่านั้น:

\[
\theta =
Quantile_{0.35}\left(\lvert S_t\rvert;\;t \in Train\right)
\]

Daily hard routing:

\[
R_t =
\begin{cases}
Bull, & S_t > \theta \\
Bear, & S_t < -\theta \\
Sideway, & -\theta \le S_t \le \theta
\end{cases}
\]

35th percentile ไม่ได้บังคับให้ outer-test ต้องมี Sideway 35%; มันกำหนด deadband จาก
training distribution เท่านั้น Test distribution สามารถเปลี่ยนตามตลาดได้จริง

Soft memberships คำนวณจากระยะถึง symmetric boundaries ด้วย temperature 0.35 และรวมได้
หนึ่งทุกแถว ค่าเหล่านี้เป็น distance-based memberships ไม่ใช่ calibrated posterior
probabilities จึงใช้ hard routing เป็น primary และเก็บ soft routing เป็น secondary
experiment

## 5. Pre-test development screen

เปรียบเทียบ Sideway quantile 0.30, 0.35 และ 0.40 โดย fit threshold บน 2012–2019 แล้วตรวจ
2020–2021 เท่านั้น

| Quantile | Validation Bull | Validation Sideway | Validation Bear | Sideway mean 20D return |
|---:|---:|---:|---:|---:|
| 0.30 | 30.58% | 36.98% | 32.44% | +0.044% |
| **0.35** | **26.65%** | **43.39%** | **29.96%** | **+0.169%** |
| 0.40 | 22.52% | 49.79% | 27.69% | +0.452% |

เลือก 0.35 เป็นค่ากลางที่รักษา Sideway sample เพียงพอสำหรับ downstream experts โดยไม่
ขยาย deadband จน validation Sideway เกือบครึ่งหนึ่งเหมือน quantile 0.40 ค่า mean 20-day
return ของ Sideway ยังอยู่ใกล้ศูนย์มากเมื่อเทียบกับ Bull +7.62% และ Bear -7.18% ใน
validation เดียวกัน

ไฟล์ `protocol_development.csv` เก็บผลทุก candidate, split และ regime

## 6. Walk-forward data

| Fold | Training period | Test period | Train rows | Test rows |
|---|---:|---:|---:|---:|
| 1 | 2012-05-03–2021-12-30 | 2022-01-04–2022-12-30 | 2,359 | 241 |
| 2 | 2012-05-03–2022-12-30 | 2023-01-03–2023-12-28 | 2,600 | 243 |
| 3 | 2012-05-03–2023-12-28 | 2024-01-02–2024-12-30 | 2,843 | 244 |
| 4 | 2012-05-03–2024-12-30 | 2025-01-02–2025-12-18 | 3,087 | 234 |

รวม outer-test 962 trading days

## 7. Outer-test daily regime distribution

| Test year | Bull | Sideway | Bear |
|---:|---:|---:|---:|
| 2022 | 73 (30.29%) | 105 (43.57%) | 63 (26.14%) |
| 2023 | 40 (16.46%) | 84 (34.57%) | 119 (48.97%) |
| 2024 | 81 (33.20%) | 104 (42.62%) | 59 (24.18%) |
| 2025 | 74 (31.62%) | 58 (24.79%) | 102 (43.59%) |
| **Pooled** | **268 (27.86%)** | **351 (36.49%)** | **343 (35.65%)** |

ผลมี face validity ตามสภาวะตลาดโดยรวม:

- 2022 ซึ่งผลตอบแทนทั้งปีใกล้ศูนย์มี Sideway มากที่สุด
- 2023 และ 2025 มี Bear มากที่สุด
- 2024 มี Bull + Sideway เป็นสัดส่วนหลัก

ตารางรายปีเป็นเพียง aggregate; router สร้างผลหนึ่งรายการต่อ trading day และเก็บ daily
features, memberships และ label ครบทุกแถว

## 8. Training semantic profiles

ช่วงค่าจากทั้ง 4 expanding folds:

| Regime | Mean 20D return | Mean ADX | Training share |
|---|---:|---:|---:|
| Bull | +3.84% ถึง +3.99% | 24.06–24.64 | 35.91–37.98% |
| Sideway | +0.127% ถึง +0.156% | 18.05–18.32 | 35.00–35.02% |
| Bear | -4.88% ถึง -4.35% | 28.52–29.39 | 27.00–29.09% |

ทุก fold ตรงตาม semantic conditions:

- Bear mean return < Sideway mean return < Bull mean return
- Sideway mean 20-day return ใกล้ศูนย์ที่สุด
- Sideway mean absolute semantic score ต่ำที่สุด
- Sideway ADX ต่ำกว่า Bull และ Bear สม่ำเสมอ
- Minimum training regime share สูงกว่า locked gate 15%

ผล: **ผ่าน quality gate 4/4 folds**

## 9. Threshold stability

| Fold | Training end | Training-only threshold |
|---|---:|---:|
| 1 | 2021-12-30 | 0.065984 |
| 2 | 2022-12-30 | 0.064101 |
| 3 | 2023-12-28 | 0.064143 |
| 4 | 2024-12-30 | 0.063138 |

Coefficient of variation = 1.85%; range เทียบ mean = 4.42% แสดงว่า deadband ไม่ได้
เปลี่ยนรุนแรงเมื่อ expanding training period เพิ่มขึ้น

### 9.1 Parameter sensitivity

ทำ robustness analysis โดยเปลี่ยนทีละ parameter และ fit threshold จาก training ของแต่ละ
fold ใหม่ทุกครั้ง ไม่มีการใช้ผลนี้ย้อนเลือก protocol:

| Variant | Agreement กับ selected v2 | ARI กับ selected v2 | Training gates |
|---|---:|---:|---:|
| Sideway quantile 0.30 | 92.18–97.86% | 0.801–0.944 | Pass 4/4 |
| Sideway quantile 0.40 | 94.65–96.58% | 0.842–0.909 | Pass 4/4 |
| EWMA span 2 | 91.36–95.73% | 0.770–0.887 | Pass 4/4 |
| EWMA span 5 | 92.12–94.87% | 0.765–0.868 | Pass 4/4 |

ไม่มี variant ใดทำให้ regime ใดหายไปจาก outer-test และทุก variant ยังผ่าน training semantic
gates ผลจึงไม่ขึ้นกับ quantile 0.35 หรือ span 3 เพียงจุดเดียว รายละเอียดราย fold อยู่ใน
`sensitivity_analysis.csv`

## 10. Causality and leakage audit

สำหรับทุก fold:

1. คำนวณ features/memberships/labels จาก train + full test
2. ตัดข้อมูลประมาณกึ่งกลาง test
3. คำนวณใหม่จาก prefix ด้วย training threshold เดิม
4. เปรียบเทียบทุกค่าบน prefix

| Fold | Prefix rows | Max feature difference | Max membership difference | Label mismatches |
|---|---:|---:|---:|---:|
| 1 | 2,479 | 0.0 | 0.0 | 0 |
| 2 | 2,721 | 0.0 | 0.0 | 0 |
| 3 | 2,965 | 0.0 | 0.0 | 0 |
| 4 | 3,204 | 0.0 | 0.0 | 0 |

Unit test เปลี่ยน target ทั้ง training และ test เป็นค่าคงที่ผิดปกติแล้ว labels ไม่เปลี่ยน
และอีก test เปลี่ยน test returns ทั้งชุดแล้ว training threshold ไม่เปลี่ยน จึงยืนยันทั้ง
target independence และ training-only calibration

## 11. HMM v1 comparison

Daily v2 agreement กับ HMM v1 บน outer tests:

| Fold/year | Agreement |
|---|---:|
| 2022 | 44.40% |
| 2023 | 53.91% |
| 2024 | 38.11% |
| 2025 | 43.16% |

agreement ต่ำเป็นผลที่คาดได้เพราะ v2 แก้ construct: HMM v1 Sideway คือ high-volatility
transition state ขณะที่ v2 Sideway คือ low-directional-strength/range-bound state ไม่ควร
กล่าวว่าวิธีใด “แม่นกว่า” หากไม่มี independent ground-truth regime labels แต่ v2 เหมาะกว่า
สำหรับคำถามวิจัยที่ต้องเลือก feature ตาม Bull/Sideway/Bear semantics

## 12. Runtime and verification

- Runtime รวม 4 folds รวม artifact, causality และ sensitivity audits:
  **3.118 seconds**
- Core selected fold-fit runtime รวม: 0.277 seconds
- Daily train/test labels ที่สร้าง: 11,851 rows
- Output: 16 CSV และ 5 JSON
- Output integrity audit: passed; ไม่พบ unexpected non-finite values, duplicate daily
  dates หรือ membership rows ที่รวมไม่เท่ากับหนึ่ง
- Daily v2 tests: 8 passed
- Statement coverage: 82%
- Integrated Track A–C regression suite: 83 passed
- Black: passed
- Ruff: passed

## 13. Reproducibility

คำสั่ง:

```powershell
py -3.12 -m models.track_c_daily_regime
```

ไฟล์สำคัญ:

- Source: `models/track_c_daily_regime.py`
- Tests: `tests/test_track_c_daily_regime.py`
- Daily labels: `outputs/track_c/daily_regime_v2/fold_1` ถึง `fold_4`
- Fold gates/runtime: `outputs/track_c/daily_regime_v2/fold_summary.csv`
- Daily distribution: `outputs/track_c/daily_regime_v2/regime_distribution.csv`
- Semantic profiles: `outputs/track_c/daily_regime_v2/semantic_profiles.csv`
- Development candidates: `outputs/track_c/daily_regime_v2/protocol_development.csv`
- Threshold drift: `outputs/track_c/daily_regime_v2/threshold_stability.csv`
- Parameter robustness: `outputs/track_c/daily_regime_v2/sensitivity_analysis.csv`
- Prefix audit: `outputs/track_c/daily_regime_v2/causality_audit.csv`
- HMM ablation comparison:
  `outputs/track_c/daily_regime_v2/hmm_baseline_comparison.csv`
- Config, versions, hashes และ limitation:
  `outputs/track_c/daily_regime_v2/run_metadata.json`

## 14. Downstream integration contract

Progressive SHAP ขั้นถัดไปต้อง:

1. ใช้ `routing_regime` ของวัน \(t\) กับ direction target ของวัน \(t+1\)
2. ภายในแต่ละ inner walk-forward split ต้อง fit \(\theta\) ใหม่จาก inner-training เท่านั้น
3. สร้าง EWMA score ต่อเนื่องจาก inner-training ไป inner-validation แบบ causal
4. เลือก features แยก Bull/Sideway/Bear เฉพาะ inner-training/validation ที่อนุญาต
5. กำหนด minimum regime sample และ global fallback ก่อนเปิด outer-test metrics
6. ใช้ hard routing เป็น primary; soft membership routing เป็น secondary ablation
7. เปรียบเทียบกับ global model และ HMM v1 router

ห้ามใช้ fold-level threshold ในไฟล์ปัจจุบันย้อนกลับไปทำ inner-validation feature selection
เพราะ threshold นั้นเห็น full outer-training distribution แล้ว ต้องเรียก
`fit_fold_daily_regimes` ใหม่ในแต่ละ inner split

## 15. Paper-ready conclusion

Daily multi-timescale semantic regime v2 สร้าง Bull/Sideway/Bear label ทุก trading day โดย
รวม risk-adjusted returns ระยะ 1–60 วันกับ ADX และ causal EWMA smoothing Sideway ถูกนิยาม
ด้วย symmetric deadband ที่ fit จาก training distribution เท่านั้น ระบบผ่าน semantic,
minimum-size, causality, stability และ software quality gates ครบทุก fold และแก้ zero-Sideway
failure ของ HMM v1 ได้โดยไม่บังคับ class distribution บน test set

protocol v2 จึงถูกเลือกเป็น regime router สำหรับ nested Progressive SHAP ส่วน HMM v1 คงไว้
เป็น ablation อย่างไรก็ตามผลปี 2022–2025 ต้องรายงานเป็น post-hoc robustness evidence เพราะ
protocol correction เริ่มขึ้นหลังเห็น semantic failure ของ HMM outer diagnostics

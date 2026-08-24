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

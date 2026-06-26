3. Leakage Prevention Rules
3.1 Daily Features

Daily features ใช้ข้อมูลของวัน t ได้ เช่น:

Open_D
High_D
Low_D
Close_D
Volume_D
Change_pct_D

เพราะถือว่าทำนายหลังตลาดปิดของวัน t

3.2 Weekly Features

Weekly features ต้องเป็นข้อมูลของ previous completed week เท่านั้น

Allowed:
Close_W = ราคาปิดของสัปดาห์ก่อนหน้า

Not allowed:
Close_W = ราคาปิดปลายสัปดาห์ปัจจุบันที่ยังไม่เกิดขึ้น ณ วัน t
3.3 Monthly Features

Monthly features ต้องเป็นข้อมูลของ previous completed month เท่านั้น

Allowed:
Close_M = ราคาปิดของเดือนก่อนหน้า

Not allowed:
Close_M = ราคาปิดสิ้นเดือนปัจจุบันที่ยังไม่เกิดขึ้น ณ วัน t
3.4 Target Leakage

ห้ามใช้ feature ที่สร้างจากอนาคต เช่น:

Target_Next_Close
Target_Direction
Next_Day_Return
Future_Return
Close_D(t+1)
feature ใด ๆ ที่คำนวณจากข้อมูลหลังวัน t
4. Feature Groups Used in This Stage

Stage นี้ยังไม่ใช้ Technical Indicators

ใช้เฉพาะกลุ่ม feature ต่อไปนี้:

Raw Multi-Timeframe OHLCV
Lag Features
Return-Based Features
Rolling Trend Features
Rolling Volatility Features
Momentum / ROC Features
Cross-Timeframe Features
Candlestick / Price Action Features
Volume Pressure Features
Direction History Features
5. Model-by-Model Feature Mapping
5.1 Ridge Regression
Input Shape
2D Array: [Samples, Features]
Feature Set

Ridge Regression ใช้ Full Non-TA Feature Pool หรือ SHAP-Selected Top 30 Features

Features Used
A. Raw D/W/M OHLCV
Open_D, High_D, Low_D, Close_D, Volume_D, Change_pct_D
Open_W, High_W, Low_W, Close_W, Volume_W, Change_pct_W
Open_M, High_M, Low_M, Close_M, Volume_M, Change_pct_M
B. Lag Features
Close_D_lag1, Close_D_lag2, Close_D_lag3, Close_D_lag5, Close_D_lag10, Close_D_lag20, Close_D_lag60
Change_pct_D_lag1, Change_pct_D_lag2, Change_pct_D_lag3, Change_pct_D_lag5, Change_pct_D_lag10, Change_pct_D_lag20
Volume_D_lag1, Volume_D_lag2, Volume_D_lag3, Volume_D_lag5, Volume_D_lag10, Volume_D_lag20
Close_W_lag1, Close_W_lag2, Close_W_lag4
Close_M_lag1, Close_M_lag3
C. Return Features
Return_1D
Return_3D
Return_5D
Return_10D
Return_20D
Return_60D
D. Rolling Trend Features
SMA_5
SMA_10
SMA_20
SMA_60
Close_to_SMA_5
Close_to_SMA_10
Close_to_SMA_20
Close_to_SMA_60
E. Rolling Volatility Features
Volatility_5
Volatility_10
Volatility_20
Volatility_60
F. Momentum / ROC Features
Momentum_5
Momentum_10
Momentum_20
Momentum_60
ROC_5
ROC_10
ROC_20
ROC_60
G. Cross-Timeframe Features
Ratio_Close_D_to_W
Ratio_Close_D_to_M
Ratio_Volume_D_to_W
H. Candlestick / Price Action Features
Spread_D
Body_D
Body_abs_D
Upper_Shadow_D
Lower_Shadow_D
Body_to_Range_D
Close_Position_D
Spread_W
Body_W
Close_Position_W
Spread_M
Body_M
Close_Position_M
I. Volume Pressure Features
Volume_MA_5
Volume_MA_20
Volume_Ratio_5
Volume_Ratio_20
Volume_Change_1D
Volume_Change_5D
Volume_Change_20D
J. Direction History Features
Direction_lag1
Direction_lag2
Direction_lag3
Direction_lag5
Up_Ratio_5
Up_Ratio_10
Up_Ratio_20
Down_Ratio_5
Down_Ratio_10
Down_Ratio_20
Scaling
ใช้ StandardScaler
Notes

Ridge เป็น linear model จึงควร scaling ทุก feature เพื่อให้ coefficient และ regularization ทำงานสมเหตุสมผล

5.2 XGBoost
Input Shape
2D Array: [Samples, Features]
Feature Set

XGBoost ใช้ Full Non-TA Feature Pool หรือ SHAP-Selected Top 30 Features

Features Used

ใช้ feature ชุดเดียวกับ Ridge Regression:

Raw D/W/M OHLCV
Lag Features
Return Features
Rolling Trend Features
Rolling Volatility Features
Momentum / ROC Features
Cross-Timeframe Features
Candlestick / Price Action Features
Volume Pressure Features
Direction History Features
Scaling
ไม่จำเป็นต้อง scaling
Notes

XGBoost เป็น tree-based model จึงรับค่าดิบได้ดี และสามารถจับ non-linear relationship ระหว่าง feature ได้

5.3 LightGBM
Input Shape
2D Array: [Samples, Features]
Feature Set

LightGBM ใช้ Full Non-TA Feature Pool หรือ SHAP-Selected Top 30 Features

Features Used

ใช้ feature ชุดเดียวกับ Ridge Regression และ XGBoost:

Raw D/W/M OHLCV
Lag Features
Return Features
Rolling Trend Features
Rolling Volatility Features
Momentum / ROC Features
Cross-Timeframe Features
Candlestick / Price Action Features
Volume Pressure Features
Direction History Features
Scaling
ไม่จำเป็นต้อง scaling
Notes

LightGBM เหมาะกับ feature pool ขนาดกลางถึงใหญ่ และสามารถใช้ร่วมกับ SHAP เพื่ออธิบาย feature importance ได้ดี

5.4 AutoGluon
Input Shape
2D DataFrame: [Samples, Features]
Feature Set

AutoGluon ใช้ Full Non-TA Feature Pool หรือ SHAP-Selected Top 30 Features

Features Used

ใช้ feature ชุดเดียวกับ tabular models อื่น:

Raw D/W/M OHLCV
Lag Features
Return Features
Rolling Trend Features
Rolling Volatility Features
Momentum / ROC Features
Cross-Timeframe Features
Candlestick / Price Action Features
Volume Pressure Features
Direction History Features
Scaling
ไม่ต้อง scaling เอง

AutoGluon สามารถจัดการ preprocessing ภายในได้บางส่วน

Fixed Baseline Setting
presets = "medium_quality"
time_limit = 120 seconds per fold
hyperparameters = "default"
no custom hyperparameter search
Notes

AutoGluon ในงานนี้ถูกจำกัดเวลาและ configuration เพื่อให้เป็น baseline ที่ควบคุมได้ ไม่ใช่การค้นหา hyperparameter อย่างละเอียด

5.5 LSTM
Input Shape
3D Tensor: [Samples, Time Steps, Features]
Sliding Window Sizes

Deep learning model ใช้ sliding windows ดังนี้:

[5, 10, 20, 40, 60]

แต่ละ window หมายถึงจำนวนวันย้อนหลังที่ LSTM เห็นก่อนทำนายวันถัดไป

ตัวอย่าง:

Window = 20

Input shape ต่อ 1 sample:
[20 trading days, 18 raw features]

Target:
Close_D(t+1) หรือ Direction(t+1)
Primary Feature Set: LSTM-Raw

LSTM-Raw ใช้เฉพาะ raw sequential features จำนวน 18 ตัว:

Daily Raw Features
Open_D
High_D
Low_D
Close_D
Volume_D
Change_pct_D
Weekly Raw Features
Open_W
High_W
Low_W
Close_W
Volume_W
Change_pct_W
Monthly Raw Features
Open_M
High_M
Low_M
Close_M
Volume_M
Change_pct_M
Optional Feature Set: LSTM-Enhanced

หากต้องการทำ ablation เพิ่ม สามารถใช้ LSTM-Enhanced ได้ โดยเพิ่มเฉพาะ selected non-TA features ที่เหมาะกับ sequence model เช่น:

Return_1D
Return_5D
Return_20D
Volatility_5
Volatility_20
Close_to_SMA_5
Close_to_SMA_20
Volume_Ratio_20
Close_Position_D
Up_Ratio_5
Up_Ratio_20

แต่ใน baseline หลัก แนะนำให้เริ่มจาก LSTM-Raw ก่อน

Scaling
ใช้ MinMaxScaler หรือ StandardScaler

กฎสำคัญ:

fit scaler บน training fold เท่านั้น
transform validation/test fold ด้วย scaler ตัวเดิม
Notes

LSTM ไม่จำเป็นต้องใช้ lag features แบบ tabular เพราะ temporal dependency ถูกสร้างผ่าน sliding window แล้ว

อย่างไรก็ตาม ไม่ควรเขียนว่า LSTM ห้ามใช้ engineered features โดยเด็ดขาด ควรเขียนว่า:

LSTM-Raw is used as the primary sequence baseline, while LSTM-Enhanced can be evaluated as an additional ablation.
5.6 Chronos Tiny Zero-Shot Greedy
Input Shape
1D Array: [Sequence Length]
Feature Set

Chronos ใช้เพียง feature เดียว:

Close_D
Context Length
64 trading days
Prediction Length
1 trading day
Input Example
[Close_D(t-63), Close_D(t-62), ..., Close_D(t)]
Output
Predicted Close_D(t+1)
Scaling
ไม่ต้อง scaling

Chronos มี tokenization / normalization ภายในโมเดล

Fixed Baseline Setting
model = chronos-t5-tiny
fine-tuning = disabled
sampling = disabled
decoding = greedy
prediction_length = 1
context_length = 64
Notes

Chronos ถูกประเมินภายใต้ strict univariate zero-shot setting จึงไม่ใช้ feature อื่น เช่น Open, High, Low, Volume, Weekly, Monthly หรือ engineered features

6. Compact Summary Table
Model	Feature Set	Feature Count	Input Shape	Scaling
Ridge Regression	Full Non-TA Feature Pool / SHAP Top 30	~100 or 30	2D	StandardScaler
XGBoost	Full Non-TA Feature Pool / SHAP Top 30	~100 or 30	2D	ไม่จำเป็น
LightGBM	Full Non-TA Feature Pool / SHAP Top 30	~100 or 30	2D	ไม่จำเป็น
AutoGluon	Full Non-TA Feature Pool / SHAP Top 30	~100 or 30	2D	ไม่จำเป็น
LSTM-Raw	Raw D/W/M OHLCV	18	3D, windows [5,10,20,40,60]	MinMaxScaler / StandardScaler
LSTM-Enhanced	Raw + selected Non-TA features	~25-35	3D, windows [5,10,20,40,60]	MinMaxScaler / StandardScaler
Chronos Tiny	Close_D only	1	1D, context 64	ไม่จำเป็น

Ridge Regression:
Full Non-TA Feature Pool -> StandardScaler -> Ridge

XGBoost:
Full Non-TA Feature Pool -> XGBoost

LightGBM:
Full Non-TA Feature Pool -> LightGBM

AutoGluon:
Full Non-TA Feature Pool -> AutoGluon medium_quality, 120 sec/fold

LSTM:
Raw D/W/M OHLCV 18 features -> Sliding windows [5,10,20,40,60] -> Scaler -> LSTM

Chronos:
Close_D only -> Context length 64 -> Chronos Tiny Zero-Shot Greedy

First Step Mapping:

Stage 1 Naive ทุกตัวใช้ Raw D/W/M OHLCV
Stage 2 Full Non-TA Feature Pool
Stage 3 Full + TA Feature Pool
Stage 4 Full + TA Feature Pool
Stage 5 Only 30 SHAP Top
Stage 6 SHAP Protocol Refinement
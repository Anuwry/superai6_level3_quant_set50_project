1. Data Cleaning: จัดการวันหยุด กลับด้านเวลา (เสร็จแล้ว)
2. Feature Engineering: สร้าง Lag, SMA, Momentum ฯลฯ จากราคาดิบๆ (เรากำลังจะทำขั้นตอนนี้)
3. Walk-Forward Split: ตัดแบ่งข้อมูล Train และ Test ทีละปี
4. Data Scaling (ทางเลือก): หากใช้โมเดล Neural Network (Deep Learning) ค่อยทำ MinMax ตอนนี้ โดย Fit ที่ Train แล้ว Transform ที่ Test (แต่ถ้าใช้ XGBoost ให้ข้ามขั้นตอนนี้ไปเลย)
5. Model Training: โยนเข้าโมเดลเพื่อทำนายผล
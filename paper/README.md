# Manuscript working package

โฟลเดอร์นี้เป็นชุดงานเขียนสำหรับบทความ SET50 reliability-audit study โดยใช้
ผลการทดลองจากโมเดลที่ลงทะเบียนไว้ 5 แบบ ได้แก่ LSTM, CNN, LSTM-CNN,
LSTM-Attention และ LSTM-CNN-Attention งานหลักนำเสนอ reliability-audit
framework ไม่ได้อ้างว่าเสนอโมเดลใหม่หรือ state-of-the-art architecture

## Current authoritative manuscript

- **Current formatted manuscript (20 August 2026):** `SET_direction_manuscript_journal_formatted_v3.docx` — 35 pages, 7 tables, 10 figures, 14 displayed equations at 12 pt, 67 live Zotero citations, concise Word heading styles, 10 pt captions/notes, paper-style inline mathematical notation, and a redesigned Figure 5 with clearer spacing. Formatting and QA details are recorded in `JOURNAL_FORMATTING_LOG_2026-08-20.md`.

- **Current graphical abstract (20 August 2026):** `output/graphical_abstract_v3/graphical_abstract_set_reliability_visual_v3.png` and `output/pdf/graphical_abstract_set_reliability_visual_v3.pdf` — an image-rich soft-minimal 2:1 workflow with no overall title or bottom message band. It uses custom mini-plots and scientific diagrams to show the market/news/model inputs, five audit dimensions and complete principal results while retaining generous spacing and a restrained blue-grey palette. Design and verification details are recorded in `GRAPHICAL_ABSTRACT_LOG_2026-08-20.md`.

- `newest_original_manuscript_results_visuals_integrated_cited.docx` — ไฟล์
  manuscript หลัก ณ วันที่ 18 สิงหาคม 2026 มี 7 ตาราง 10 รูป 14 สมการ และ
  live Zotero citations 67 จุด
- `CITATION_COMPLETION_LOG_2026-08-18.md` — บันทึกการเติมและตรวจ Zotero
  citations รอบล่าสุด
- `RESULT_VISUALS_INTEGRATION_LOG_2026-08-18.md` — บันทึกการนำ prediction,
  SHAP และ LIME visualizations เข้า Results
- `table_and_figure_plan.md` — แผนและแหล่งข้อมูลของตารางและรูปทั้งหมด
- `supplementary_material_v1.md` — ร่าง Supplementary Materials
- `references.bib` — BibTeX library ของโครงการ

ไฟล์ DOCX ชื่ออื่นในโฟลเดอร์นี้เป็น intermediate versions สำหรับตรวจย้อนกลับ
และไม่ควรใช้แทนไฟล์ authoritative ข้างต้นโดยอัตโนมัติ

## Work timeline (Asia/Bangkok)

| วันที่ | งานที่ทำ | ผลลัพธ์/สถานะ | ความรู้และทักษะที่พัฒนา | ปัญหาที่พบและวิธีแก้ |
|---|---|---|---|---|
| 22 มิถุนายน 2026 | เริ่ม repository และวาง pipeline สำหรับการทำนายทิศทาง SET50 วันถัดไป | กำหนด target, expanding-window structure และโครงสร้างข้อมูลเริ่มต้น | เข้าใจการแปลงโจทย์ราคาหุ้นเป็น next-day direction forecasting และการออกแบบ time-series pipeline | ปัญหาแรกคือโจทย์ยังไม่ชัดว่าต้องทำนายระดับราคาหรือทิศทาง และการแบ่งข้อมูลทั่วไปอาจนำข้อมูลอนาคตย้อนเข้าไปในการฝึก จึงแก้ด้วยการนิยามให้ข้อมูล ณ วัน t ทำนายวัน t+1 และวาง expanding-window evaluation ตามลำดับเวลาไว้ตั้งแต่ต้น |
| 23 มิถุนายน 2026 | ปรับลำดับการเตรียมข้อมูลและตรวจการเชื่อมต่อข้อมูลรายวัน | ได้ขั้นตอน preprocessing รุ่นแรกสำหรับใช้พัฒนาต่อ | พัฒนาความรู้เรื่องการเรียงข้อมูลตามเวลา การตรวจวันซื้อขาย และการป้องกันข้อมูลผิดลำดับ | พบว่าการเชื่อมข้อมูลที่ใช้เพียงลำดับแถวอาจผิดเมื่อมีวันหยุดหรือวันที่ขาดหาย จึงจัดเรียงตามวันที่ ตรวจวันซ้ำและวันขาด และใช้วันซื้อขายจริงเป็นกุญแจเชื่อมข้อมูลแทนการสมมติว่าทุกวันมีหนึ่งแถว |
| 24 มิถุนายน 2026 | เตรียม raw market data ให้พร้อมสำหรับ pipeline | ได้ข้อมูลตลาดเวอร์ชันพร้อมตรวจและประมวลผล | พัฒนาทักษะด้าน data cleaning, schema checking และการรักษา raw data ไว้ตรวจย้อนกลับ | รูปแบบคอลัมน์ ชนิดข้อมูล และช่วงวันที่ของไฟล์ดิบไม่เหมือนกันทั้งหมด จึงแยก raw data ออกจาก processed data ทำ schema checks แปลงวันที่และตัวเลขอย่างชัดเจน และเก็บต้นฉบับไว้เพื่อย้อนตรวจเมื่อผลผิดปกติ |
| 25 มิถุนายน 2026 | ปรับปรุงชุดข้อมูลและ feature preparation รุ่นล่าสุด | ได้ dataset version ที่ใช้เป็นฐานของการทดลองถัดไป | เรียนรู้เรื่อง dataset versioning, feature alignment และความสำคัญของ reproducible inputs | เมื่อปรับข้อมูลหลายรอบเกิดความเสี่ยงว่าจะไม่ทราบว่าโมเดลแต่ละรอบใช้ไฟล์ใด จึงแก้ด้วยการแยกเวอร์ชัน dataset กำหนดชื่อ output ให้สื่อความหมาย และรักษาลำดับ feature ให้ตรงกันระหว่าง train, validation และ test |
| 26 มิถุนายน 2026 | รัน naive/model baselines รุ่นแรก | เก็บผล RMSE, MAE และ Direction Accuracy เพื่อใช้ตรวจ sanity ของระบบ | เข้าใจความต่างระหว่างการทายระดับราคาใกล้เคียงกับการทายทิศทางถูก และบทบาทของ sanity baseline | ผลบางแบบให้ค่า error ของระดับราคาดูดีแต่ทาย Up/Down ได้ใกล้การสุ่ม ทำให้เห็นว่า R² หรือ RMSE เพียงอย่างเดียวตอบโจทย์ไม่ได้ จึงเพิ่ม Direction Accuracy และต่อมาล็อก Balanced Accuracy เป็นตัวชี้วัดทิศทางหลัก |
| 1 กรกฎาคม 2026 | ทดลอง Full non-TA feature set และสำรวจ Optuna tuning | เก็บเป็น development evidence และไม่ใช้การปรับจูนหลังเห็น outer-test result เป็นข้ออ้างหลัก | พัฒนาความรู้เรื่อง hyperparameter optimization, overfitting จาก model selection และขอบเขตการใช้ Optuna อย่างเป็นธรรม | ปัญหาคือ Optuna สามารถค้นหาค่าที่ดูดีจากข้อมูลเดิมจนเกิด selection overfitting ได้ จึงจำกัดการปรับจูนไว้ใน development data แยก outer test ออกจากการเลือกค่า และไม่ใช้ผลที่ปรับหลังเห็นปี 2022–2025 เป็นหลักฐานยืนยัน |
| 17 กรกฎาคม 2026 | รัน Full-TA และ persistence checks | ได้ numerical baseline และ technical-analysis feature pool สำหรับการเปรียบเทียบภายหลัง | เข้าใจ technical indicators, persistence behavior และการสร้าง comparator ที่ตรวจสอบได้ | การเปรียบเทียบโมเดลจะไม่ยุติธรรมหากแต่ละตัวใช้ feature pool ต่างกัน จึงสร้าง Full-TA comparator กลาง ตรวจ persistence behavior และกำหนดชุดข้อมูลเดียวกันเพื่อให้ความแตกต่างมาจาก architecture หรือ treatment ที่กำลังทดสอบจริง |
| 28 กรกฎาคม 2026 | ทำ causal rolling VMD, sweep neural windows 1, 3, 5, 10 และ 20 วัน ปิด Track A และเริ่ม Track B news-data pilot | Track A ครบ 400 fits พร้อม runtime; ระบุช่องว่างข่าว 2012–2025 และขอบเขตข่าวที่รายงานได้จริง | พัฒนาความรู้เรื่อง causal denoising, VMD, sliding-window selection, paired ablation, multi-seed evaluation และ news-data provenance | พบความเสี่ยงสำคัญว่าการทำ VMD ทั้งอนุกรมก่อนแบ่ง train/test จะส่งข้อมูลอนาคตเข้าสู่อดีต และ window ที่ดีที่สุดอาจถูกเลือกจาก test set จึงเปลี่ยนเป็น rolling VMD ที่ใช้ย้อนหลัง 60 วันเท่านั้น เลือก neural window จากช่วงก่อนปี 2022 และรัน paired seeds พร้อมเก็บ runtime ส่วนข่าวพบว่าไม่มี coverage ที่สม่ำเสมอตลอด 2012–2025 จึงบันทึก gap และจำกัด claim ให้ตรงกับข้อมูลที่มีจริง |
| 30 กรกฎาคม 2026 | ปิด Track B | ได้ expanding out-of-sample Local-NLP sentiment, locked 2023 Single-pass/Leader benchmark, frozen 2024–2025 news extension และ paired market/news fusion | เข้าใจ out-of-sample sentiment construction, multimodal fusion และความต่างระหว่าง intrinsic sentiment accuracy กับ downstream forecasting value | ปัญหาคือข่าวจำนวนมากไม่มี sentiment label และการใช้ label ที่สร้างจากโมเดลเดียวกันในช่วง train/test อาจทำให้รั่วไหล จึงสร้าง annual expanding out-of-sample predictions สำหรับปีที่มี label แล้ว freeze ตัวจำแนกสำหรับข่าว 2024–2025 พร้อมแยกการวัด sentiment accuracy ออกจากคำถามว่าข่าวช่วยพยากรณ์ตลาดหรือไม่ |
| 31 กรกฎาคม 2026 | ทำ causal Bull/Sideway/Bear regimes, Track C SHAP/LIME และ Track D partial-2026 | ปิด progressive SHAP selection, LIME fidelity audit, outer inference, forward stress test, economic proxy และ XAI sanity checks | พัฒนาความรู้เรื่อง regime labeling, train-only feature selection, SHAP/LIME reliability, distribution shift และการไม่ตีความ exploratory backtest เป็นกำไรที่ยืนยันแล้ว | Regime รุ่นแรกเสี่ยงมี Sideway น้อยเกินไปและการเลือก feature จากข้อมูลทั้งหมดจะรั่วไหล ขณะที่ LIME บางครั้งอธิบายโมเดลได้ไม่ตรง จึงสร้าง causal multi-horizon trend score ใช้ deadband จาก training fold เท่านั้น คำนวณ SHAP และเลือก top-k ภายใน train และรายงาน LIME low-fidelity ทุกครั้งแทนการตัดทิ้ง ส่วนผล 2026 และ economic proxy ถูกลดสถานะเป็น stress test และ exploratory evidence |
| 1 สิงหาคม 2026 | รวมข่าวเข้ากับ regime-SHAP pipeline บน common cohort | ปิด post-hoc 2×2 Global/Regime-SHAP × Numeric/+News experiment; API cost เพิ่ม USD 0 และไม่มี BAcc contrast ผ่าน Holm correction | เข้าใจ factorial ablation, common-cohort comparison, multimodal interaction และ multiple-testing control | ก่อนหน้านี้ Track B กับ Track C ถูกทดสอบแยกกันจึงยังตอบไม่ได้ว่าข่าวทำงานร่วมกับ regime-selected features อย่างไร จึงสร้างการทดลอง 2×2 บนวันที่ร่วมกัน ใช้ architecture, window, seed และ budget เดียวกัน และแยก LLM intrinsic benchmark ออกจาก Local-NLP downstream features เพื่อไม่ให้สรุปเกินหลักฐาน |
| 3 สิงหาคม 2026 | Freeze SET100 same-exchange transfer และเริ่ม Strong-Q2 hardening | กำหนด protocol ก่อนเปิดผล SET100 โดยคง features, windows, seeds และ training budget จาก SET50 | พัฒนาความรู้เรื่อง protocol freezing, transportability, same-exchange robustness และ external-validity boundaries | ปัญหาคือหากปรับโมเดลใหม่บน SET100 ผลจะไม่ใช่การทดสอบ transfer ของระบบ SET50 อีกต่อไป และ SET100 ก็ไม่ใช่ตลาดอิสระจาก SET50 จึง freeze ทุกค่าก่อนรัน ห้าม retune และกำหนดคำเรียกอย่างตรงไปตรงมาว่า same-exchange breadth transfer ไม่ใช่ external-market replication |
| 4 สิงหาคม 2026 | ปิด SET100 transfer, compute-matched LLM audit, market-data governance, multimodal falsification และ public replication package พร้อมทดสอบ PIT-CMM-LSTM และ PIT-DERN | SET100 ครบ 100 fits; Leader เหนือ compute-matched controls; เพิ่ม falsification controls 400 fits; สร้าง package แบบ fail-closed และ exploratory models ทั้งสองไม่ผ่านเกณฑ์ promote | พัฒนาความรู้เรื่อง compute-matched controls, Holm correction, data governance, reproducibility, secret protection, promotion gates และการรายงาน negative results อย่างโปร่งใส | พบข้อกังขาหลายด้านพร้อมกัน ได้แก่ Leader ใช้จำนวน API calls มากกว่า single pass ข่าวอาจเป็นเพียง random extra features การทดสอบจำนวนมากเพิ่ม false positives และข้อมูลตลาดสาธารณะไม่ได้แปลว่าเปิดแจกซ้ำได้ จึงเพิ่ม equal-call และ near-cost controls เพิ่ม News-Only, shuffled, lagged และ random controls ใช้ Holm กับ block bootstrap บันทึก provenance/timezone/adjustment และสร้าง public package ที่กัน raw rows, predictions, checkpoints และ keys ออกโดยอัตโนมัติ |
| 5 สิงหาคม 2026 | ปิด PIT-FCG-LSTM และจัดทำ candidate shortlist | PIT-FCG-LSTM ไม่ผ่าน inner promotion gate จึงไม่เปิด outer evaluation และไม่ถูกนำไปแทน frozen five-model benchmark | เรียนรู้การใช้ predeclared promotion gate, novelty collision audit และการหยุดพัฒนาโมเดลก่อน outer test เมื่อหลักฐาน development ไม่สนับสนุน | แนวคิด gating มีส่วนประกอบคล้ายงานเดิมจำนวนมากและผล development ไม่ชนะเกณฑ์ที่กำหนด การนำไปลอง outer test ซ้ำจะเพิ่มโอกาสเลือกโมเดลจากความบังเอิญ จึงทำ novelty collision audit ใช้ inner gate ที่ freeze ล่วงหน้า และหยุดโมเดลทันทีเมื่อไม่ผ่าน |
| 6 สิงหาคม 2026 | ออกแบบและทดสอบ PIT-TLDN พร้อม novelty/SOTA review และเริ่ม PIT-SET50-CRIN | PIT-TLDN ผ่าน development gate บางส่วน แต่ debate output ไม่ชนะ strongest worker จึงไม่ promote | พัฒนาความรู้เรื่อง worker/leader architecture, component ablation และเหตุผลที่ ensemble หรือ debate ไม่ได้ดีกว่าองค์ประกอบที่ดีที่สุดเสมอไป | แม้โมเดล debate รวม worker หลายแบบแล้วดูซับซ้อนขึ้น แต่ผลรวมไม่ชนะ worker ที่ดีที่สุดและ CNN worker ไม่ยืนยันความเชี่ยวชาญด้าน trend ตามสมมติฐาน จึงตรวจ component ablation แยก worker/leader และเลือกไม่ promote แทนการอ้างว่าความซับซ้อนคือ novelty หรือประสิทธิภาพที่เหนือกว่า |
| 7 สิงหาคม 2026 | ปิด PIT-SET50-CRIN, TCRC-LSTM, PIT-CDR-LSTM, SEA-LSTM และ FCTA-LSTM พร้อมเริ่ม supplementary package | ทุก candidate ถูกปิดตาม frozen failure rules; ไม่มีการ tune หรือ rerun เพื่อบีบผล และเก็บ negative screens เป็น audit trail | พัฒนาทักษะด้าน fair common-cohort benchmarking, failure-rule discipline, retrospective evidence labeling และการแยกผล exploratory ออกจาก main claim | การพัฒนา Ours model หลังเห็นผลปี 2024–2025 มีความเสี่ยงสูงต่อ post-selection bias และบางการทดลองเดิมใช้ cohort ที่เทียบกับห้าโมเดลไม่ได้ จึงบังคับ common dates, fixed seeds, fixed budget และ one-shot protocols แล้วปิดทุก candidate ที่แพ้โดยไม่แก้ architecture หรือรันใหม่เพื่อไล่ผล |
| 11 สิงหาคม 2026 | สร้าง manuscript draft, ตรวจ references/citation metadata และทำ Figure 1 | ได้ manuscript v1, citation audit, Zotero additions และ reliability-audit pipeline figure | พัฒนาความรู้เรื่อง academic narrative, source verification, citation management และการออกแบบ research figure แบบเรียบง่าย | พบ citation keys ที่ไม่มีรายการจริงใน library และรูป pipeline รุ่นแรกดูซับซ้อนหรือเหมือนภาพ AI มากเกินไป จึงตรวจชื่อเรื่อง ผู้แต่ง ปี DOI และแหล่งเผยแพร่ของแต่ละ reference ก่อนเพิ่มเข้า Zotero พร้อมออกแบบ Figure 1 ใหม่ให้เป็นกล่อง ลูกศร และสีที่เรียบง่ายตามงานวารสาร |
| 13 สิงหาคม 2026 | ปรับโครงสร้าง manuscript | ย่อ Methods เหลือ 8 ส่วน ย้ายผลลัพธ์ออกจาก Methods เพิ่ม Conclusion, Limitations, Future work, Acknowledgements และ Reproducibility/Data Availability พร้อมแก้ spacing | เรียนรู้โครงสร้างบทความวารสาร การแยกวิธีวิจัยออกจากผล และการเขียน limitations อย่างไม่ลดทอน contribution | Methods เดิมมีหัวข้อย่อยมากเกินไปและบางย่อหน้ารายงานผลลัพธ์ปะปนอยู่ ทำให้ narrative อ่านยาก จึงรวมเนื้อหาที่เกี่ยวข้องให้เหลือแปดส่วน ย้าย empirical findings ไป Results และเพิ่มส่วนท้ายที่วารสารมักต้องการโดยคงสาระสำคัญเดิม |
| 14 สิงหาคม 2026 | เพิ่มและจัดหมายเลข Tables, Figures และ Equations | ตัด Table 3C ที่ไม่มี authoritative common-cohort artifact และปรับ Figure 5/figure-table mapping | พัฒนาทักษะด้าน evidence-to-table mapping, artifact reconciliation, equation numbering และการไม่รายงานตารางที่หลักฐานยังไม่ครบ | Table 3C ยังไม่มี common-cohort result ที่ตรวจสอบได้ และหมายเลขรูป/ตารางบางจุดไม่ตรงกับตำแหน่งในเนื้อหา จึงตัดตารางและข้อความ claim ที่เกี่ยวข้องออกจาก main manuscript แล้วสร้าง mapping ให้ทุก Figure และ Table เชื่อมกับหลักฐานที่มีจริง |
| 15 สิงหาคม 2026 | ปรับสมการและข้อความอธิบายเชิงวิธีวิจัย | จัดสมการหลักและนิยามตัวแปรให้ต่อเนื่องกับ Methods | พัฒนาความรู้เรื่อง mathematical notation, equation referencing และการอธิบายสูตรให้ตรวจสอบซ้ำได้ | สมการบางชุดใช้หมายเลขย่อยและนิยามสัญลักษณ์ไม่ครบ ทำให้ผู้อ่านตามวิธีคำนวณได้ยาก จึงเปลี่ยนเป็นหมายเลขต่อเนื่อง 1–14 เพิ่มคำอธิบายตัวแปรและเชื่อมสมการกับ citation ของวิธีต้นทาง |
| 17 สิงหาคม 2026 | แยก Results กับ Discussion และเพิ่มการเปรียบเทียบกับงานใกล้เคียง | Results รายงานหลักฐานก่อน ส่วน Discussion อธิบายกลไก ข้อจำกัด และความสัมพันธ์กับวรรณกรรม | พัฒนาความรู้เรื่อง analytical discussion, claim–evidence separation และการใช้วรรณกรรมอธิบายผลบวก ผลลบ และผลที่ไม่ชัดเจน | เนื้อหาเดิมรวมผลและการตีความจนไม่ชัดว่าส่วนใดเป็นข้อมูลที่วัดได้กับส่วนใดเป็นคำอธิบาย จึงให้ Results เริ่มจากตัวเลข ตาราง และ uncertainty ก่อน แล้วเก็บการสังเคราะห์ภาพรวมไว้ Discussion ขณะเดียวกันยังใส่ citation ใกล้ผลเพื่อเทียบกับงานที่เกี่ยวข้องโดยไม่เปลี่ยน inference ของเรา |
| 18 สิงหาคม 2026 | ปรับ prediction plots และ SHAP/LIME visualization วางลง Results และปิด citation audit | เพิ่ม scatter, 2025 time-series, explainability audit; แปลง citation 11 จุดเป็น live Zotero fields และตรวจครบ 35 หน้า 7 ตาราง 10 รูป | พัฒนาทักษะด้าน diagnostic visualization, model-behavior interpretation, Zotero field management และ visual/structural QA ของ Word manuscript | กราฟเดิมมีหัวข้อและคำอธิบายทับกัน อีกทั้งการเกาะเส้นราคาอาจทำให้เข้าใจผิดว่าโมเดลทายทิศทางเก่ง จึงลดองค์ประกอบที่ไม่จำเป็น แยก scatter กับ time-series แสดง RMSE/MAE ควบคู่ BAcc และอภิปราย turning points กับ Down recall ส่วน citation ที่ยังเป็นตัวหนาหรือ plain text ถูกแปลงเป็น live Zotero fields แล้ว render ตรวจทุกหน้า |
| 19 สิงหาคม 2026 | อัปเดต README และ work timeline | ระบุไฟล์ authoritative รวมหลักฐานการทำงานและงานที่เหลือไว้ในจุดเดียว | พัฒนาทักษะด้าน research documentation, project audit trail และการสรุปความก้าวหน้าสำหรับอาจารย์หรือผู้ประเมิน | ไฟล์ผลทดลอง log และ manuscript มีหลายเวอร์ชันจนเสี่ยงหยิบผิดหรือเล่า timeline ไม่ตรงกัน จึงกำหนด authoritative manuscript ให้ชัด รวมวันทำงาน ผล ความรู้ ปัญหาและวิธีแก้ไว้ใน README เดียว และระบุว่าไฟล์อื่นเป็น intermediate artifacts สำหรับตรวจย้อนหลัง |
| 20 สิงหาคม 2026 | จัดรูปแบบ manuscript สำหรับวารสารและออกแบบ Figure 5 ใหม่ | ได้ไฟล์ authoritative v3 จำนวน 35 หน้า โดยคง 7 ตาราง 10 รูป 14 สมการ และ live Zotero citations 67 จุด; Figure 5 แยกขั้นตอนของ forecasting audit กับ intrinsic LLM audit ชัดขึ้น | พัฒนาทักษะด้าน Word style hierarchy, mathematical typography, caption formatting, vector-like academic diagram design และ full-document visual QA | Figure 5 รุ่นก่อนมีกล่องและลูกศรชิดกันเมื่อย่อให้พอดีกับหน้ากระดาษ จึงจัดตำแหน่งใหม่ให้มีช่องว่างระหว่าง source, processing, model และ endpoint เพิ่มขึ้น รักษาอัตราส่วนเดิมเพื่อไม่ให้ pagination เปลี่ยน และ render ตรวจครบทั้ง 35 หน้า พร้อมตรวจว่า Zotero fields และโครงสร้างเอกสารยังสมบูรณ์ |

## ความรู้ที่พัฒนาและความหมายของสิ่งที่ได้เรียนรู้

ในช่วงเริ่มต้นได้เรียนรู้ว่า repository คือพื้นที่กลางที่รวบรวม code, configuration,
ผลการทดลอง และเอกสารของโครงการไว้ด้วยกันเพื่อให้ติดตามการเปลี่ยนแปลงได้ ส่วน
pipeline คือชุดขั้นตอนที่พาข้อมูลตั้งแต่ไฟล์ดิบไปจนถึงผลพยากรณ์ เช่น การอ่านข้อมูล
การตรวจความถูกต้อง การสร้าง feature การแบ่ง train/test การฝึกโมเดล และการวัดผล
คำว่า preprocessing จึงหมายถึงการเตรียมข้อมูลก่อนส่งเข้าโมเดล ไม่ว่าจะเป็นการ
เรียงวันที่ ลบข้อมูลซ้ำ แปลงชนิดตัวเลข จัดการ missing values หรือปรับสเกล ขณะที่
schema checking คือการตรวจว่าชื่อคอลัมน์ ชนิดข้อมูล หน่วย และโครงสร้างไฟล์ตรงตาม
ข้อตกลงที่ระบบต้องการ การแยก raw data ออกจาก processed data ช่วยให้ข้อมูลต้นฉบับ
ไม่ถูกแก้ทับ และ dataset versioning คือการระบุว่าการทดลองแต่ละรอบใช้ชุดข้อมูล
เวอร์ชันใด ทำให้สามารถย้อนกลับมาสร้างผลเดิมได้ ส่วน feature alignment คือการ
รับประกันว่าคอลัมน์และลำดับของตัวแปรที่ส่งเข้า train, validation และ test ตรงกัน

คำว่า time series หมายถึงข้อมูลที่ลำดับเวลาก่อนและหลังมีความสำคัญ ต่างจากข้อมูล
ตารางทั่วไปที่สามารถสลับแถวได้อย่างอิสระ โจทย์ next-day forecasting ในงานนี้คือ
การใช้ข้อมูลที่ทราบได้เมื่อสิ้นสุดวัน t เพื่อทำนายว่าระดับ SET50 ในวันซื้อขาย t+1
จะสูงหรือต่ำกว่าวัน t ส่วน expanding window คือการจำลองสถานการณ์จริงโดยเริ่มจาก
ข้อมูลอดีตก้อนหนึ่งแล้วเพิ่มข้อมูลใหม่เข้า training set เมื่อเวลาเดินไปข้างหน้า
ขณะที่ sliding window ในงานนี้หมายถึงจำนวนวันย้อนหลังที่ถูกส่งเป็นหนึ่ง sequence
เข้า neural network เช่น 1, 3, 5, 10 หรือ 20 วัน Label-date purging คือการตัดแถว
ฝึกที่คำตอบของแถวนั้นจะถูกเปิดเผยหลังจากวันเริ่มต้นของชุดประเมินออก แม้ feature
ของแถวดังกล่าวจะอยู่ในอดีตก็ตาม วิธีนี้ป้องกันไม่ให้โมเดลเรียนรู้คำตอบที่ในเวลา
จริงยังไม่เกิดขึ้น

Temporal leakage คือการที่ข้อมูลจากอนาคตเข้าสู่ขั้นตอนฝึกหรือเลือกโมเดลโดยตรง
หรือโดยอ้อม ตัวอย่างเช่น การ fit scaler ด้วยข้อมูลทุกปี การทำ VMD ทั้งอนุกรมก่อน
แบ่งชุด การเลือก feature จาก test set หรือเลือก window จากผลปีที่ต้องการรายงาน
ปัญหานี้ทำให้ผลดูดีกว่าความสามารถที่ใช้งานจริง Point-in-time protocol จึงเป็น
กติกาที่กำหนดว่าทุก feature, label, scaler, decomposition และ model decision ต้อง
ใช้เฉพาะข้อมูลที่มีอยู่ ณ เวลานั้น Common cohort หมายถึงการเปรียบเทียบโมเดลบน
วันที่ตัวอย่างและ target ชุดเดียวกัน Paired ablation หมายถึงการเปลี่ยนเพียง
องค์ประกอบที่กำลังศึกษา เช่น มีหรือไม่มี VMD โดยคง architecture, seed, fold และ
training budget เหมือนเดิม ส่วน multi-seed evaluation คือการฝึกซ้ำด้วยค่าเริ่มต้น
สุ่มหลายค่าเพื่อแยกความผันผวนจากการฝึกออกจากความเปลี่ยนแปลงข้ามปี และ runtime คือ
เวลาที่ใช้สร้าง ฝึก และประเมินโมเดล ซึ่งช่วยวัดต้นทุนการคำนวณควบคู่กับความแม่นยำ

ในด้าน architecture ได้เข้าใจว่า LSTM หรือ Long Short-Term Memory เป็น recurrent
neural network ที่ใช้ประตูกำหนดว่าข้อมูลส่วนใดควรถูกจำ ลืม หรือส่งต่อ จึงเหมาะกับ
ความสัมพันธ์ที่ต่อเนื่องตามเวลา CNN หรือ Convolutional Neural Network ใช้ filter
เลื่อนไปตาม sequence เพื่อค้นหารูปแบบระยะสั้น เช่น การเร่งตัว การกลับทิศ หรือรูปทรง
เฉพาะบริเวณ Attention เป็นกลไกที่เรียนรู้ว่าส่วนใดของ sequence ควรได้รับน้ำหนัก
มากกว่า ส่วน hybrid architecture คือการนำกลไกหลายชนิดมาต่อกัน เช่น LSTM-CNN หรือ
LSTM-CNN-Attention โดยหวังให้แต่ละส่วนช่วยกัน แต่จำนวนพารามิเตอร์และความยืดหยุ่น
ที่เพิ่มขึ้นอาจทำให้ optimization ยากขึ้น เกิด variance สูง หรือจำรูปแบบเฉพาะชุด
ฝึกมากเกินไปได้ จึงต้องพิสูจน์ด้วย ablation ไม่สามารถสรุปจากความซับซ้อนของชื่อ
โมเดลเพียงอย่างเดียว

Baseline คือจุดอ้างอิงพื้นฐานที่ใช้ตรวจว่าระบบซับซ้อนสร้างประโยชน์จริงหรือไม่ เช่น
การคงค่าราคาเดิม การใช้แบบจำลองเรียบง่าย หรือโมเดลที่ไม่มี treatment เพิ่มเติม
RMSE คือรากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง จึงลงโทษข้อผิดพลาดขนาดใหญ่
มากเป็นพิเศษ MAE คือค่าเฉลี่ยของขนาดความผิดพลาดและตีความในหน่วยราคาได้ตรงกว่า
Direction Accuracy คือสัดส่วนวันที่ทาย Up/Down ถูกทั้งหมด แต่จะดูสูงเกินจริงได้
เมื่อคลาสไม่สมดุล Balanced Accuracy จึงเฉลี่ย recall ของ Up และ Down ให้น้ำหนัก
สองฝั่งเท่ากัน ส่วน MCC หรือ Matthews Correlation Coefficient สรุปคุณภาพของ
confusion matrix ทั้งสี่ช่องและมีค่าใกล้ศูนย์เมื่อ prediction แทบไม่สัมพันธ์กับ
คำตอบจริง Class-wise recall ช่วยเปิดเผยว่าโมเดลทายฝั่งใดได้ดีหรือแย่ และ coverage
บอกสัดส่วนตัวอย่างที่โมเดลมี prediction ที่นำไปประเมินได้

Optuna คือ framework สำหรับค้นหา hyperparameters โดยลองชุดค่าหลายแบบและใช้ผล
validation นำทางการค้นหา แม้ช่วยลดแรงงาน แต่หากทดลองมากเกินไปหรือใช้ test set
เป็นตัวตัดสินก็สามารถเกิด selection overfitting ได้ Inner development จึงหมายถึง
ช่วงข้อมูลที่ใช้พัฒนาและเลือกค่า ส่วน outer test คือข้อมูลที่กันไว้ประเมินผลหลัง
การตัดสินใจเสร็จ Promotion gate คือเกณฑ์ที่กำหนดก่อนดูผลว่าโมเดลใหม่ต้องดีขึ้น
เท่าใดและสม่ำเสมอกี่ช่วงเวลาจึงจะนำไปทดสอบต่อ Failure rule คือกติกาว่าเมื่อไม่
ผ่านต้องหยุดโดยไม่ปรับหลังเห็นผล และ novelty collision audit คือการค้นวรรณกรรม
เพื่อดูว่าส่วนประกอบหรือแนวคิดที่เรียกว่าใหม่มีคนใช้แล้วหรือไม่ วิธีเหล่านี้ช่วย
ลดการสร้าง Ours model ด้วยการทดลองซ้ำหลายครั้งจนบังเอิญชนะข้อมูลที่ถูกเปิดดูแล้ว

VMD ย่อมาจาก Variational Mode Decomposition เป็นวิธีแยกสัญญาณหนึ่งเส้นออกเป็น
องค์ประกอบย่อยที่มีช่วงความถี่แตกต่างกัน โดยแต่ละองค์ประกอบเรียกว่า mode หรือ
intrinsic mode function งานนี้มอง mode ที่มี centre frequency สูงสุดเป็นส่วนที่
มีความถี่สูงและทดลองนำออกเพื่อสร้าง denoised close คำว่า causal rolling VMD
หมายความว่าการแยกสัญญาณของวัน t ใช้เฉพาะราคาย้อนหลังถึงวัน t ไม่ใช้ข้อมูลหลังวัน
นั้น ส่วน denoising คือการลดความผันผวนที่คาดว่าเป็น noise แต่การทดลองทำให้เข้าใจ
ว่าความผันผวนความถี่สูงบางส่วนอาจมีข้อมูลเกี่ยวกับการกลับทิศวันถัดไป การลด RMSE
จึงไม่จำเป็นต้องเพิ่ม Balanced Accuracy

Market regime คือการแบ่งสภาพตลาดออกเป็นช่วงที่มีพฤติกรรมต่างกัน Bull หมายถึง
ภาวะที่แนวโน้มขึ้นเด่น Bear หมายถึงแนวโน้มลงเด่น และ Sideway หมายถึงช่วงที่ยัง
ไม่มีทิศทางชัด งานนี้ไม่ได้ใช้คำตอบในอนาคตมาตั้ง regime แต่สร้าง causal trend
score จากผลตอบแทนหลายช่วงเวลา ความผันผวนย้อนหลัง และ ADX ซึ่งเป็นดัชนีวัดความ
แข็งแรงของแนวโน้ม จากนั้นใช้ deadband ที่เรียนจาก training fold เป็นขอบเขตแยก
สามสถานะ วิธีนี้ทำให้ regime เป็น feature ที่ทราบได้ก่อนการทำนายจริง ไม่ใช่ label
ที่สร้างย้อนหลังจากราคาซึ่งเกิดขึ้นแล้ว

SHAP หรือ SHapley Additive exPlanations เป็นวิธีคำนวณว่าน้ำหนักของ feature แต่ละ
ตัวมีส่วนผลัก prediction ออกจากค่าฐานมากน้อยเพียงใด แนวคิดมาจาก Shapley value ที่
เฉลี่ย contribution ของผู้เล่นเมื่อเข้าร่วมชุดในลำดับต่าง ๆ งานนี้ใช้ค่าเฉลี่ย
absolute SHAP ภายใน training fold เพื่อจัดอันดับ feature ไม่ได้ใช้เพื่อกล่าวว่า
feature นั้นเป็นสาเหตุของการขึ้นหรือลงของตลาด ส่วน LIME หรือ Local Interpretable
Model-agnostic Explanations เป็นการสร้างข้อมูลรบกวนรอบตัวอย่างหนึ่งแล้ว fit โมเดล
เรียบง่ายให้เลียนแบบ prediction ในบริเวณนั้น คำว่า local fidelity คือระดับที่
โมเดลเรียบง่ายเลียนแบบโมเดลจริงได้ ซึ่งวัดด้วย R² ในงานนี้ หาก fidelity ต่ำ
คำอธิบายของ LIME จะถูกเก็บเป็นข้อจำกัดและไม่นำไปสร้างเรื่องเล่าเกี่ยวกับ feature

ในส่วนข่าวและภาษา ได้พัฒนาความเข้าใจเกี่ยวกับการสร้าง sentiment แบบ
out-of-sample ซึ่งหมายถึง sentiment ของแต่ละปีต้องมาจากตัวจำแนกที่ไม่เคยเห็น
label ของปีนั้น Local-NLP คือแบบจำลองภาษาที่รันภายในเครื่อง ในงานนี้ใช้ character
TF-IDF ซึ่งแปลงลำดับตัวอักษรเป็นค่าน้ำหนักตามความถี่ร่วมกับ logistic regression
เพื่อทำนาย positive, neutral หรือ negative จากนั้น daily aggregation คือการรวม
ข่าวหลายชิ้นที่ถูกจับคู่กับวันซื้อขายเดียวกันให้เป็นค่าเฉลี่ย ส่วนเบี่ยงเบน สัดส่วน
แต่ละ sentiment และจำนวนข่าว การทดสอบ Bull/Bear/Leader เป็น role-structured LLM
inference ที่ให้ worker ฝั่งบวกและลบสร้างเหตุผลก่อนให้ Leader ตัดสิน Self-
consistency คือการเรียกโมเดลซ้ำด้วยโจทย์เดียวแล้วรวมคำตอบ Equal-call control ใช้
จำนวนครั้งเท่ากัน ส่วน near-cost control ใช้ต้นทุนใกล้กัน เพื่อแยกประโยชน์ของ
โครงสร้างบทบาทออกจากประโยชน์ที่เกิดเพียงเพราะใช้ inference budget มากขึ้น
Intrinsic sentiment accuracy คือความถูกต้องของการจำแนกข่าวเอง ขณะที่ downstream
forecasting gain คือการที่ sentiment ช่วยเพิ่มความแม่นยำของการพยากรณ์ตลาดหลัง
รวมกับ market features สองค่านี้เป็นคนละคำถามและใช้แทนกันไม่ได้

ด้านสถิติและความน่าเชื่อถือของงานวิจัย ได้เรียนรู้การเฉลี่ย seed ภายใน
model-year ก่อนนำปีมาเป็นหน่วยอนุมาน Exact sign-flip test คือการสลับเครื่องหมาย
ของ paired effect ในแต่ละปีครบทุกความเป็นไปได้ แล้วตรวจว่าค่าเฉลี่ยที่สังเกตได้
สุดโต่งเพียงใดภายใต้สมมติฐานไม่มีผล Holm adjustment คือวิธีปรับ p-value ของการ
ทดสอบหลายรายการเป็นลำดับเพื่อลดโอกาสเกิด false positive อย่างน้อยหนึ่งรายการใน
ครอบครัวการทดสอบ Moving-block bootstrap คือการสุ่มข้อมูลเป็นบล็อกวันที่ต่อเนื่อง
แทนการสุ่มทีละวัน เพื่อรักษาความสัมพันธ์ตามเวลาหรือ serial dependence ไว้ใน
ตัวอย่างจำลอง Confidence interval คือช่วงค่าที่สื่อความไม่แน่นอนของ effect ส่วน
statistical significance เป็นเกณฑ์ตัดสินภายใต้แบบทดสอบที่กำหนด การไม่มีผลที่ผ่าน
เกณฑ์ไม่ได้พิสูจน์ว่า effect เป็นศูนย์ แต่หมายความว่าข้อมูลและกำลังการทดสอบปัจจุบัน
ยังไม่พอที่จะสรุปอย่างมั่นใจ

Partial forward test คือการนำโมเดลที่ freeze แล้วไปวัดกับช่วงเวลาหลัง development
แต่ยังมีข้อมูลไม่ครบทั้งปี จึงใช้เป็น stress test มากกว่าหลักฐาน prospective แบบ
เต็มปี Frozen transfer คือการนำระบบเดิมไปใช้กับข้อมูลอีกชุดโดยไม่ปรับค่าใหม่
เพื่อดูว่าสิ่งที่เรียนรู้เคลื่อนย้ายได้หรือไม่ Distribution shift หมายถึงการที่
การกระจายของ feature, label หรือแหล่งข้อมูลเปลี่ยนจากช่วงฝึก Transportability คือ
ความสามารถของข้อสรุปหรือโมเดลที่จะยังใช้ได้เมื่อเปลี่ยนประชากรหรือบริบท SET100
ในงานนี้จึงทดสอบความทนทานเมื่อขยายดัชนีภายในตลาดเดียวกัน แต่ไม่ถือเป็น external
replication เพราะยังอยู่ในตลาด ช่วงเวลา และเศรษฐกิจมหภาคเดียวกับ SET50

สุดท้าย โครงการนี้ได้พัฒนาทักษะด้านวินัยการวิจัยและการสื่อสารเชิงวิชาการอย่างมาก
ทั้งการ freeze protocol ก่อนดูผล การตั้ง promotion และ failure gates สำหรับ
โมเดลใหม่ การยอมเก็บ negative results แทนการปรับโมเดลซ้ำจนชนะ test set การบันทึก
provenance ซึ่งหมายถึงหลักฐานว่าข้อมูลมาจากที่ใด ดาวน์โหลดเมื่อใด และผ่านขั้นตอน
อะไร Provider terms คือเงื่อนไขของเจ้าของแหล่งข้อมูล Timezone และ information
cutoff กำหนดเวลาอ้างอิงและเวลาสุดท้ายที่ feature ถือว่าทราบได้ ส่วน adjustment
convention ระบุว่าราคาเป็นระดับที่ผู้ให้บริการเผยแพร่หรือผ่านการปรับ split,
dividend หรือ corporate action แบบใด Replication package คือชุด code, protocol,
environment และ aggregate outputs ที่ผู้อื่นใช้ตรวจงานได้ โดย fail-closed หมายถึง
หากไม่แน่ใจว่าไฟล์ปลอดภัยต่อการเผยแพร่ ระบบจะไม่รวมไฟล์นั้นไว้ก่อน จึงไม่เปิดเผย
raw restricted data, row-level predictions, private checkpoints หรือ credentials

ในด้านการเขียน Methods คือส่วนที่อธิบายว่าทำอะไรและทำอย่างไร Results คือส่วนที่
รายงานสิ่งที่วัดได้พร้อมตาราง รูป และความไม่แน่นอน ส่วน Discussion คือส่วนที่
ตีความว่าเหตุใดผลจึงเป็นเช่นนั้น เปรียบเทียบกับวรรณกรรม และกำหนดขอบเขตของข้อสรุป
Claim–evidence mapping คือการตรวจว่าทุกข้อความสำคัญมี artifact หรืองานอ้างอิงรองรับ
Zotero คือโปรแกรมจัดการรายการอ้างอิงที่เก็บ metadata และสร้าง citation/bibliography
ตามรูปแบบวารสาร Live Zotero field หมายถึง citation ใน Word ที่ยังเชื่อมกับ Zotero
และเปลี่ยน style หรือ refresh ได้ ส่วน visual QA คือการ render เอกสารเป็นภาพหรือ
PDF แล้วตรวจทุกหน้าว่าตาราง รูป สมการ และย่อหน้าไม่ทับ ตัด หรือเลื่อนผิดตำแหน่ง
สิ่งที่ได้จากโครงการจึงไม่ใช่เพียงผลเปรียบเทียบห้าโมเดล แต่เป็นความเข้าใจว่า
งานพยากรณ์ตลาดที่น่าเชื่อถือต้องควบคุมข้อมูล เวลา งบการคำนวณ สถิติ การเผยแพร่
หลักฐาน และขอบเขตของข้อสรุปไปพร้อมกัน

## Evidence boundaries retained in the manuscript

- Table 3C ถูกตัดออกแล้ว เพราะยังไม่มี authoritative Leader-derived
  downstream common-cohort artifact
- Downstream news features มาจาก frozen expanding Local-NLP pipeline ส่วน
  Bull/Bear/Leader เป็น locked intrinsic sentiment benchmark ที่รายงานแยกกัน
- SET100 เป็น same-exchange breadth/transfer audit ไม่ใช่ independent external
  market replication
- Partial-2026 เป็น source-contingent stress test ไม่ใช่ full-year prospective
  validation
- Exploratory proposed-model candidates ที่ไม่ผ่าน frozen gates ไม่ถูกนำไปแทน
  five-model benchmark หรือใช้เป็น headline success claim

## Remaining submission tasks

1. เลือกวารสารและใช้ Word template/reference style ตาม Guide for Authors
2. ยืนยันชื่อผู้เขียน affiliations, ORCID และ corresponding-author information
3. เติม Funding, CRediT author contributions และ Conflict of Interest จากข้อมูลจริง
4. ฝาก reproducibility package แล้วแทนข้อความ placeholder ด้วย repository URL/DOI
5. เตรียม Graphical Abstract, Highlights และ Cover Letter หากวารสารกำหนด
6. ตรวจภาษา, citation punctuation และ cross-references รอบสุดท้ายหลัง Zotero Refresh

## Reproduction notes

Track A five-model pipeline:

```powershell
py -3.12 -m models.run_five_model_pipeline all --force
```

Integrated multimodal experiment:

```powershell
python -m models.integrated_multimodal_runner run
python -m models.integrated_multimodal_runner aggregate
```

Market-data governance:

```powershell
$env:PYTHONPATH=(Get-Location).Path
D:\conda_envs\my_env\python.exe scripts\run_market_data_governance.py
```

Public package and manuscript evidence:

```powershell
py -3.12 scripts/build_manuscript_artifacts.py
py -3.12 scripts/build_public_replication_package.py
py -3.12 scripts/audit_public_replication_package.py
```

Raw SET50/SET100 rows, point-in-time fold CSVs, row-level predictions, private
checkpoints และ credentials ไม่รวมอยู่ใน public submission package

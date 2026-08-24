import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import pptxgen from 'pptxgenjs';
import { chromium } from 'playwright-core';

const root = 'D:/SET50_direction_prediction_paper';
const scratch = 'C:/Users/narak/AppData/Local/Temp/codex-presentations/manual-20260820/set50-progress-deck/tmp';
const htmlPath = path.join(root, 'paper/tools/set50_reliability_progress_slides.html');
const slidesDir = path.join(scratch, 'slides');
const outputDir = path.join(root, 'outputs/presentation_progress_v1');
const outputPath = path.join(outputDir, 'SET50_reliability_framework_progress_presentation.pptx');
const montagePath = path.join(outputDir, 'SET50_reliability_framework_progress_montage.png');
const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

const notes = [
  'งานนี้ไม่ได้เสนอเพียงโมเดลที่ชนะครั้งเดียว แต่สร้างกรอบตรวจสอบความน่าเชื่อถือของการพยากรณ์ทิศทางดัชนี SET ในวันถัดไป โดยรวมข้อมูลตลาด ข่าว การแบ่งภาวะตลาด และการทดสอบการถ่ายโอนผลลัพธ์ไว้ในกระบวนการเดียว',
  'เริ่มจากปัญหาหลัก การทำนายระดับราคาให้ใกล้เคียงไม่ได้แปลว่าจะทายทิศทางขึ้นหรือลงถูกต้อง เพราะสัญญาณรายวันมีขนาดเล็กและเปลี่ยนตามเวลา อีกทั้งผลที่ดูดีอาจเกิดจากการเลือกช่วงเวลา โมเดล หรือข้อมูลที่ไม่เป็น point-in-time',
  'สิ่งที่พัฒนาคือ reliability framework ที่ตรวจห้ามิติภายใต้กติกาเดียวกัน ได้แก่ข้อมูลตามเวลาจริง การลดสัญญาณรบกวน ข่าวและ LLM การอธิบายแบบแยกภาวะตลาด และการทดสอบช่วงเวลาใหม่กับ SET100 แต่ละส่วนใช้โมเดลและช่วงทดสอบที่จับคู่กันเพื่อให้เปรียบเทียบได้',
  'ข้อมูล SET50 ครอบคลุมปี 2012 ถึง 2025 และใช้ปี 2022 ถึง 2025 เป็น outer test รวม 962 วันซื้อขาย การสร้าง feature scaler การเลือก feature และการแปลงข้อมูลทั้งหมดเรียนรู้จาก train เท่านั้น ส่วน label ของวันถัดไปถูก purged ตามวันที่ข้อมูลจะสังเกตได้จริง',
  'Track ตัวเลขทดสอบ LSTM CNN LSTM-CNN LSTM-Attention และ LSTM-CNN-Attention ร่วมกับ causal rolling VMD ภายใต้ frozen windows ผลต่าง balanced accuracy อยู่ระหว่างลบ 0.60 ถึงบวก 0.35 จุด จึงสรุปได้ว่า VMD ไม่ได้เพิ่มผลอย่างสม่ำเสมอทุกสถาปัตยกรรม',
  'งานข่าวถูกแยกเป็นสองคำถาม ฝั่งซ้ายถามว่าคะแนนข่าวที่ทำนายนอกตัวอย่างเพิ่มผลการพยากรณ์ SET50 หรือไม่ และมี falsification controls ฝั่งขวาวัดความสามารถ sentiment ของระบบ Bull Bear Leader โดยตรง Leader ดีกว่า compute-matched controls ราวหกจุด แต่ผลนี้เป็น intrinsic endpoint ไม่ใช่หลักฐานว่าการพยากรณ์ตลาดดีขึ้นโดยอัตโนมัติ',
  'ภาวะ Bull Sideway และ Bear ถูกสร้างแบบ causal แล้วใช้ SHAP เลือก feature แยกตามภาวะ ผลช่วย CNN เพิ่ม 1.46 จุด แต่มีผลผสมหรือเป็นลบในโมเดลอื่น และไม่มีผลใดผ่าน Holm adjustment ส่วน LIME มี local fidelity ต่ำใน 71.83 เปอร์เซ็นต์ของการทำซ้ำ จึงใช้เป็น stress test มากกว่า claim หลัก',
  'เมื่อรวมขั้นตอนที่ล็อกไว้ ผลเฉลี่ย balanced accuracy สูงสุดคือ LSTM-CNN-Attention ที่ 53.64 เปอร์เซ็นต์ รองลงมาคือ LSTM-CNN และ LSTM-Attention ตารางนี้สรุปจากสี่ปีทดสอบและห้า seeds จุดสำคัญคือผลเหนือ 50 เปอร์เซ็นต์มีอยู่ แต่ความแตกต่างระหว่างโมเดลมีขนาดไม่มาก',
  'กราฟปี 2025 ทำให้เห็นกลไกที่คะแนนรวมซ่อนไว้ LSTM ติดตามระดับราคาได้ใกล้กว่าในบางช่วง CNN ให้เส้นที่เรียบและเกาะแนวโน้มมากกว่า ขณะที่ hybrid อาจมีความคลาดเคลื่อนของระดับราคาสูงแม้ directional score ใกล้กัน ดังนั้นต้องแยกการประเมิน level tracking กับ direction classification',
  'ณ ตอนนี้งานวิจัย การทดลอง และเนื้อหา manuscript หลักเสร็จเรียบร้อยแล้วครับ ขั้นที่กำลังทำอยู่คือปรับ graphical abstract รอบสุดท้ายให้สรุป input วิธีการ และผลลัพธ์ได้ครบ พร้อมตรวจขนาดและรูปแบบให้ตรงตามข้อกำหนดของวารสาร หลังจากนั้นจะตรวจความสอดคล้องของชื่อรูป ตาราง ตัวเลข reference และไฟล์ supplementary อีกครั้ง แล้วจัด submission package เพื่อส่ง journal ขั้นตอนที่เหลือจึงเป็นงานเตรียมตีพิมพ์ ไม่ใช่การเพิ่มการทดลองใหม่ครับ'
];

async function makeMontage(files) {
  const montageHtml = path.join(scratch, 'montage.html');
  const cards = files.map((f, i) => `<div><img src="${pathToFileURL(f).href}"><span>${String(i + 1).padStart(2, '0')}</span></div>`).join('');
  const html = `<!doctype html><style>*{box-sizing:border-box}html,body{margin:0;width:1280px;height:720px;background:#dadde0;font-family:Arial;overflow:hidden}.grid{padding:20px;display:grid;grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(5,1fr);gap:12px;width:1280px;height:720px}.grid div{position:relative;background:white;overflow:hidden;border:1px solid #aeb4bb}.grid img{width:100%;height:100%;object-fit:contain}.grid span{position:absolute;right:6px;bottom:4px;background:#ff6b35;color:white;padding:2px 6px;font-size:12px;font-weight:bold}</style><div class="grid">${cards}</div>`;
  await fs.writeFile(montageHtml, html, 'utf8');
  const browser = await chromium.launch({ executablePath: edgePath, headless: true, args: ['--no-sandbox', '--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1.5 });
  await page.goto(pathToFileURL(montageHtml).href, { waitUntil: 'networkidle' });
  await page.screenshot({ path: montagePath });
  await browser.close();
}

async function main() {
  await fs.mkdir(slidesDir, { recursive: true });
  await fs.mkdir(outputDir, { recursive: true });

  const browser = await chromium.launch({ executablePath: edgePath, headless: true, args: ['--no-sandbox', '--disable-gpu', '--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1.5 });
  const slidePngs = [];
  for (let i = 1; i <= 10; i += 1) {
    const png = path.join(slidesDir, `slide-${String(i).padStart(2, '0')}.png`);
    const url = `${pathToFileURL(htmlPath).href}?s=${i}`;
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.screenshot({ path: png });
    slidePngs.push(png);
  }
  await browser.close();

  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'Arsanchai S.';
  pptx.company = 'Walailak University';
  pptx.subject = 'SET next-day direction reliability framework';
  pptx.title = 'Evaluating Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting';
  pptx.lang = 'en-US';
  pptx.theme = {
    headFontFace: 'Arial',
    bodyFontFace: 'Arial',
    lang: 'en-US'
  };
  pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333333, height: 7.5 });
  pptx.layout = 'CUSTOM_WIDE';

  for (let i = 0; i < slidePngs.length; i += 1) {
    const slide = pptx.addSlide();
    slide.background = { color: 'FFFFFF' };
    slide.addImage({ path: slidePngs[i], x: 0, y: 0, w: 13.333333, h: 7.5, altText: `SET50 reliability framework slide ${i + 1}` });
    slide.addNotes(notes[i]);
  }
  await pptx.writeFile({ fileName: outputPath, compression: true });
  await makeMontage(slidePngs);
  console.log(outputPath);
  console.log(montagePath);
  console.log(slidesDir);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

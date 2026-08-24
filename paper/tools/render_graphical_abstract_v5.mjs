import { pathToFileURL } from 'node:url';
import { chromium } from 'file:///C:/Users/narak/AppData/Local/Temp/codex-presentations/manual-20260820/set50-progress-deck/tmp/node_modules/playwright-core/index.mjs';

const root = 'D:/SET50_direction_prediction_paper';
const htmlPath = `${root}/paper/tools/graphical_abstract_set_reliability_v5.html`;
const pngPath = `${root}/paper/assets/graphical_abstract_set_reliability_v5.png`;
const pdfPath = `${root}/paper/assets/graphical_abstract_set_reliability_v5.pdf`;
const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

const browser = await chromium.launch({
  executablePath: edgePath,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--allow-file-access-from-files']
});

const page = await browser.newPage({
  viewport: { width: 2400, height: 960 },
  deviceScaleFactor: 1
});

await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle' });
await page.screenshot({ path: pngPath, fullPage: false });
await page.pdf({
  path: pdfPath,
  width: '25in',
  height: '10in',
  printBackground: true,
  margin: { top: '0', right: '0', bottom: '0', left: '0' }
});

await browser.close();
console.log(pngPath);
console.log(pdfPath);

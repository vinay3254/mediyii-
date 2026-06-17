/**
 * Medical AI Computer Vision Platform
 * app.js — Core application logic, tool simulators, canvas processing
 * Uses: OpenCV-style algorithms simulated in JS Canvas API
 * All 13 tools implemented with realistic demo pipelines
 */

'use strict';

// ═══════════════════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════════════════

const App = {
  activeCategory: 'all',
  webcamStreams: {},     // toolId → MediaStream
  animFrames: {},       // toolId → requestAnimationFrame ID
  chartBuffers: {},     // toolId → ring buffer of values
  rppgBuffer: [],
  respBuffer: [],
  earBuffer: [],
  rppgFPS: 30,
  lastBPM: '--',
  lastRR: '--',
  lastEAR: 1.0,
  drowsyFrameCount: 0,
};

// ═══════════════════════════════════════════════════════════════
// UTILITY HELPERS
// ═══════════════════════════════════════════════════════════════

function rand(min, max) { return Math.random() * (max - min) + min; }
function randInt(min, max) { return Math.floor(rand(min, max + 1)); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function animateBar(el, targetWidth, delay = 0) {
  if (!el) return;
  setTimeout(() => {
    el.style.width = '0%';
    el.style.transition = 'none';
    requestAnimationFrame(() => {
      el.style.transition = 'width 1.2s cubic-bezier(0.4,0,0.2,1)';
      el.style.width = targetWidth;
    });
  }, delay);
}

function setRiskDisplay(cardId, score, label) {
  const card = document.getElementById(cardId);
  if (!card) return;
  const fill = card.querySelector('.risk-fill');
  const badge = card.querySelector('.risk-badge');
  if (fill) {
    fill.style.width = score + '%';
    fill.className = 'risk-fill ' + label.toLowerCase();
  }
  if (badge) {
    badge.textContent = label;
    badge.className = 'risk-badge ' + label.toLowerCase();
  }
}

function showResults(cardId) {
  const panel = document.getElementById(cardId + '-results');
  if (panel) {
    panel.style.display = 'block';
    panel.style.opacity = '0';
    panel.style.transform = 'translateY(8px)';
    panel.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    requestAnimationFrame(() => {
      panel.style.opacity = '1';
      panel.style.transform = 'translateY(0)';
    });
  }
}

function formatNum(n, decimals = 2) {
  return typeof n === 'number' ? n.toFixed(decimals) : n;
}

// ═══════════════════════════════════════════════════════════════
// CANVAS IMAGE PROCESSING HELPERS
// ═══════════════════════════════════════════════════════════════

function getImageData(canvas) {
  const ctx = canvas.getContext('2d');
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const diff = max - min;
  let h = 0, s = max === 0 ? 0 : diff / max, v = max;
  if (diff !== 0) {
    if (max === r)      h = ((g - b) / diff + 6) % 6;
    else if (max === g) h = (b - r) / diff + 2;
    else                h = (r - g) / diff + 4;
    h /= 6;
  }
  return [h * 360, s * 100, v * 100];
}

function rgbToLab(r, g, b) {
  // sRGB → linear
  let R = r / 255, G = g / 255, B = b / 255;
  R = R > 0.04045 ? Math.pow((R + 0.055) / 1.055, 2.4) : R / 12.92;
  G = G > 0.04045 ? Math.pow((G + 0.055) / 1.055, 2.4) : G / 12.92;
  B = B > 0.04045 ? Math.pow((B + 0.055) / 1.055, 2.4) : B / 12.92;
  // → XYZ (D65)
  let X = R * 0.4124 + G * 0.3576 + B * 0.1805;
  let Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
  let Z = R * 0.0193 + G * 0.1192 + B * 0.9505;
  X /= 0.95047; Y /= 1.0; Z /= 1.08883;
  const f = v => v > 0.008856 ? Math.cbrt(v) : (7.787 * v + 16 / 116);
  const L = 116 * f(Y) - 16;
  const a = 500 * (f(X) - f(Y));
  const bv = 200 * (f(Y) - f(Z));
  return [L, a, bv];
}

function computeColorStats(imageData) {
  const d = imageData.data;
  const len = d.length / 4;
  let rSum = 0, gSum = 0, bSum = 0;
  let rSq = 0, gSq = 0, bSq = 0;
  let labB = 0;
  let counted = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    rSum += r; gSum += g; bSum += b;
    rSq += r*r; gSq += g*g; bSq += b*b;
    const [L, a, blab] = rgbToLab(r, g, b);
    labB += blab;
    counted++;
  }
  const n = counted;
  const rMean = rSum / n, gMean = gSum / n, bMean = bSum / n;
  const rStd = Math.sqrt(rSq / n - rMean * rMean);
  const gStd = Math.sqrt(gSq / n - gMean * gMean);
  const bStd = Math.sqrt(bSq / n - bMean * bMean);
  const labBMean = labB / n;
  return { rMean, gMean, bMean, rStd, gStd, bStd, labBMean };
}

// Simple Sobel edge magnitude on a canvas
function sobelMagnitude(ctx, w, h) {
  const d = ctx.getImageData(0, 0, w, h).data;
  let totalMag = 0, count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      const tl = d[((y-1)*w+(x-1))*4];
      const tm = d[((y-1)*w+ x  )*4];
      const tr = d[((y-1)*w+(x+1))*4];
      const ml = d[(y    *w+(x-1))*4];
      const mr = d[(y    *w+(x+1))*4];
      const bl = d[((y+1)*w+(x-1))*4];
      const bm = d[((y+1)*w+ x  )*4];
      const br = d[((y+1)*w+(x+1))*4];
      const gx = -tl - 2*ml - bl + tr + 2*mr + br;
      const gy = -tl - 2*tm - tr + bl + 2*bm + br;
      totalMag += Math.sqrt(gx*gx + gy*gy);
      count++;
    }
  }
  return totalMag / count;
}

// Draw animated scan line on canvas
function drawScanOverlay(ctx, w, h, progress, color = 'rgba(61,139,255,0.6)') {
  const y = Math.floor(progress * h);
  ctx.clearRect(0, 0, w, h);
  const grad = ctx.createLinearGradient(0, y - 40, 0, y + 5);
  grad.addColorStop(0, 'transparent');
  grad.addColorStop(1, color);
  ctx.fillStyle = grad;
  ctx.fillRect(0, y - 40, w, 45);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, y); ctx.lineTo(w, y);
  ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════
// MINI CHART ENGINE (draws waveform on canvas)
// ═══════════════════════════════════════════════════════════════

function MiniChart(canvasId, color, bufSize = 120) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const buf = new Array(bufSize).fill(0);
  let ptr = 0;

  function push(val) {
    buf[ptr % bufSize] = val;
    ptr++;
    draw();
  }

  function draw() {
    const w = canvas.offsetWidth || 300;
    const h = canvas.offsetHeight || 80;
    canvas.width = w; canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    const vals = [];
    for (let i = 0; i < bufSize; i++) vals.push(buf[(ptr + i) % bufSize]);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color.replace(')', ', 0.25)').replace('rgb', 'rgba'));
    grad.addColorStop(1, color.replace(')', ', 0.0)').replace('rgb', 'rgba'));

    ctx.beginPath();
    for (let i = 0; i < bufSize; i++) {
      const x = (i / (bufSize - 1)) * w;
      const y = h - ((vals[i] - mn) / range) * h * 0.85 - h * 0.075;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < bufSize; i++) {
      const x = (i / (bufSize - 1)) * w;
      const y = h - ((vals[i] - mn) / range) * h * 0.85 - h * 0.075;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  return { push, draw };
}

// ═══════════════════════════════════════════════════════════════
// 1. SKIN LESION MELANOMA CLASSIFIER
// ═══════════════════════════════════════════════════════════════

async function analyzeLeison(cardId) {
  const card = document.getElementById(cardId);
  if (!card) return;
  const input = card.querySelector('input[type="file"]');
  const btn = card.querySelector('.btn-analyze');
  const canvasOut = card.querySelector('.analysis-canvas');
  const origImg = card.querySelector('.preview-img');

  let imgSrc = origImg && origImg.src && !origImg.src.endsWith('#') ? origImg.src : null;
  if (!imgSrc && input && input.files[0]) {
    imgSrc = URL.createObjectURL(input.files[0]);
  }

  btn.innerHTML = '<div class="spinner"></div> Analyzing ABCDE features…';
  btn.classList.add('loading');
  await sleep(2200);

  // Simulated ABCDE features (realistic values)
  const hasImage = !!imgSrc;
  const asymmetry  = hasImage ? rand(0.18, 0.72) : rand(0.05, 0.30);
  const border     = hasImage ? rand(1.4, 3.8)  : rand(1.1, 1.8);
  const colorStdL  = hasImage ? rand(12, 45)    : rand(5, 18);
  const colorStdA  = hasImage ? rand(8, 30)     : rand(3, 12);
  const colorStdB  = hasImage ? rand(10, 35)    : rand(4, 15);
  const diameter   = hasImage ? rand(4, 12)     : rand(2, 6);
  const contrast   = hasImage ? rand(120, 480)  : rand(40, 130);
  const homogen    = hasImage ? rand(0.2, 0.65) : rand(0.6, 0.92);

  // Decision tree simulation (max_depth=5, ABCDE features)
  let riskScore;
  if (asymmetry > 0.45 && border > 2.5 && colorStdA > 18) riskScore = rand(72, 95);
  else if (asymmetry > 0.30 || border > 2.0) riskScore = rand(35, 70);
  else riskScore = rand(5, 30);

  const riskLevel = riskScore > 65 ? 'High' : riskScore > 35 ? 'Medium' : 'Low';

  // Draw annotated output on canvas
  if (canvasOut) {
    const ctx = canvasOut.getContext('2d');
    canvasOut.width = 280; canvasOut.height = 210;

    if (imgSrc) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, 280, 210);
        drawLesionAnnotations(ctx, 280, 210, riskLevel, asymmetry, border);
      };
      img.src = imgSrc;
    } else {
      // Generate synthetic lesion visualization
      drawSyntheticLesion(ctx, 280, 210, riskLevel);
    }
  }

  // Update UI
  setRiskDisplay(cardId, Math.round(riskScore), riskLevel);

  const featureEls = {
    'asymmetry': asymmetry, 'border': border, 'colorL': colorStdL,
    'colorA': colorStdA, 'diameter': diameter, 'contrast': contrast
  };

  for (const [key, val] of Object.entries(featureEls)) {
    const el = card.querySelector(`[data-feature="${key}"]`);
    if (el) el.textContent = formatNum(val, 2);
    const bar = card.querySelector(`[data-bar="${key}"]`);
    if (bar) {
      const pct = clamp(val / (key === 'contrast' ? 500 : key === 'border' ? 5 : key === 'diameter' ? 15 : 50) * 100, 0, 100);
      animateBar(bar, pct + '%');
    }
  }

  showResults(cardId);
  btn.innerHTML = '🔬 Re-analyze';
  btn.classList.remove('loading');

  // Decision tree text
  updateDecisionTree(cardId, { asymmetry, border, colorStdA, riskLevel, riskScore });
  updateImportanceChart(cardId, { asymmetry, border, colorStdL, colorStdA, colorStdB, diameter, contrast, homogen });
}

function drawSyntheticLesion(ctx, w, h, risk) {
  // Background skin tone
  const skinGrad = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, Math.max(w,h)/1.5);
  skinGrad.addColorStop(0, '#d4956a');
  skinGrad.addColorStop(1, '#c27a50');
  ctx.fillStyle = skinGrad;
  ctx.fillRect(0, 0, w, h);

  // Lesion
  const cx = w/2, cy = h/2;
  const rx = risk === 'High' ? 55 : risk === 'Medium' ? 45 : 35;
  const ry = risk === 'High' ? 48 : risk === 'Medium' ? 40 : 35;

  ctx.save();
  ctx.translate(cx, cy);
  if (risk === 'High') ctx.rotate(0.3);

  const lesionGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
  if (risk === 'High') {
    lesionGrad.addColorStop(0, '#1a0500');
    lesionGrad.addColorStop(0.5, '#3d1200');
    lesionGrad.addColorStop(0.8, '#5a1f00');
    lesionGrad.addColorStop(1, 'rgba(90,31,0,0)');
  } else if (risk === 'Medium') {
    lesionGrad.addColorStop(0, '#2a1500');
    lesionGrad.addColorStop(0.6, '#5a3010');
    lesionGrad.addColorStop(1, 'rgba(90,48,16,0)');
  } else {
    lesionGrad.addColorStop(0, '#6b3f20');
    lesionGrad.addColorStop(0.7, '#8b5e35');
    lesionGrad.addColorStop(1, 'rgba(139,94,53,0)');
  }

  ctx.beginPath();
  if (risk === 'High') {
    // Irregular shape
    ctx.save();
    for (let i = 0; i < 16; i++) {
      const angle = (i / 16) * Math.PI * 2;
      const jitter = 1 + rand(-0.25, 0.25);
      const x = Math.cos(angle) * rx * jitter;
      const y = Math.sin(angle) * ry * jitter;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.restore();
  } else {
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  }
  ctx.fillStyle = lesionGrad;
  ctx.fill();
  ctx.restore();

  drawLesionAnnotations(ctx, w, h, risk);
}

function drawLesionAnnotations(ctx, w, h, risk, asymmetry, border) {
  const cx = w/2, cy = h/2;
  const color = risk === 'High' ? '#ff4444' : risk === 'Medium' ? '#f59e0b' : '#10b981';

  // Bounding box
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(cx - 60, cy - 50, 120, 100);
  ctx.setLineDash([]);

  // Asymmetry axis
  ctx.strokeStyle = 'rgba(61,139,255,0.8)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(cx, cy - 55); ctx.lineTo(cx, cy + 55); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx - 65, cy); ctx.lineTo(cx + 65, cy); ctx.stroke();
  ctx.setLineDash([]);

  // Risk label
  ctx.fillStyle = color;
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`RISK: ${risk.toUpperCase()}`, 8, 18);

  // Corner brackets
  const bx = cx - 60, by = cy - 50, bw = 120, bh = 100, bSize = 12;
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  [[bx, by, 1, 1], [bx+bw, by, -1, 1], [bx, by+bh, 1, -1], [bx+bw, by+bh, -1, -1]].forEach(([x, y, dx, dy]) => {
    ctx.beginPath(); ctx.moveTo(x + dx * bSize, y); ctx.lineTo(x, y); ctx.lineTo(x, y + dy * bSize); ctx.stroke();
  });
}

function updateDecisionTree(cardId, { asymmetry, border, colorStdA, riskLevel, riskScore }) {
  const treeEl = document.getElementById(cardId + '-tree');
  if (!treeEl) return;
  const a = formatNum(asymmetry, 3), b = formatNum(border, 2), c = formatNum(colorStdA, 1);
  treeEl.innerHTML = `<span class="rule-key">|--- asymmetry</span> &lt;= 0.450
|   <span class="rule-key">|--- border_score</span> &lt;= 2.000
|   |   <span class="rule-leaf">|--- class: BENIGN (prob: ${(100-riskScore).toFixed(0)}%)</span>
|   <span class="rule-key">|--- border_score</span> &gt; 2.000
|   |   <span class="rule-key">|--- color_A_std</span> &lt;= 18.0
|   |   |   <span class="rule-leaf">|--- class: SUSPICIOUS (prob: 62%)</span>
|   |   <span class="rule-key">|--- color_A_std</span> &gt; 18.0
|   |   |   <span class="rule-leaf">|--- class: MELANOMA (prob: ${riskScore.toFixed(0)}%)</span>
<span class="rule-key">|--- asymmetry</span> &gt; 0.450
|   <span class="rule-key">|--- diameter</span> &lt;= 6.0mm
|   |   <span class="rule-leaf">|--- class: SUSPICIOUS (prob: 58%)</span>
|   <span class="rule-key">|--- diameter</span> &gt; 6.0mm
|   |   <span class="rule-leaf">|--- class: <span class="rule-val">MELANOMA (prob: ${riskScore.toFixed(0)}%)</span></span>

<span style="color:var(--text-muted)">Current sample path:</span>
asymmetry=${a} → ${asymmetry > 0.45 ? 'RIGHT' : 'LEFT'}
border=${b} → ${border > 2.0 ? 'RIGHT' : 'LEFT'}
color_A_std=${c} → <span class="rule-val">${riskLevel.toUpperCase()}</span>`;
}

function updateImportanceChart(cardId, features) {
  const names = ['asymmetry', 'border', 'L_std', 'A_std', 'B_std', 'diameter', 'contrast', 'homogen'];
  const vals   = [features.asymmetry/1, features.border/5, features.colorStdL/50, features.colorStdA/50,
                  features.colorStdB/50, features.diameter/15, features.contrast/500, 1-features.homogen];
  const maxVal = Math.max(...vals);
  const colors = ['#3d8bff','#a855f7','#ec4899','#f59e0b','#10b981','#14b8a6','#6366f1','#00d4ff'];
  const chart = document.getElementById(cardId + '-importance');
  if (!chart) return;
  chart.innerHTML = names.map((n, i) => {
    const pct = clamp((vals[i] / maxVal) * 100, 0, 100);
    return `<div class="importance-row">
      <span class="importance-name">${n}</span>
      <div class="importance-track">
        <div class="importance-fill" style="width:0%;background:${colors[i]}" data-target="${pct.toFixed(0)}%"></div>
      </div>
      <span class="importance-val">${(vals[i]*100).toFixed(0)}%</span>
    </div>`;
  }).join('');
  requestAnimationFrame(() => {
    chart.querySelectorAll('.importance-fill').forEach(el => {
      el.style.transition = 'width 1.2s cubic-bezier(0.4,0,0.2,1)';
      el.style.width = el.dataset.target;
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// 2. DIABETIC RETINOPATHY DETECTOR
// ═══════════════════════════════════════════════════════════════

async function analyzeDR(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Running blob detection…';
  btn.classList.add('loading');
  await sleep(2500);

  const maCount = randInt(0, 35);
  const heCount = randInt(0, 20);
  const grade = maCount === 0 ? 0 : maCount < 5 ? 1 : maCount < 15 ? 2 : maCount < 30 ? 3 : 4;
  const gradeLabels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
  const gradeColors = ['var(--accent-green)', 'var(--accent-teal)', 'var(--accent-amber)', '#f97316', 'var(--accent-red)'];

  // Draw fundus visualization
  if (canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = 280; canvas.height = 210;
    drawFundus(ctx, 280, 210, maCount, heCount, grade);
  }

  // Update grade display
  const gradeEl = card.querySelector('[data-grade]');
  if (gradeEl) {
    gradeEl.textContent = grade;
    gradeEl.style.color = gradeColors[grade];
  }
  const gradeLabelEl = card.querySelector('[data-grade-label]');
  if (gradeLabelEl) {
    gradeLabelEl.textContent = gradeLabels[grade];
    gradeLabelEl.style.color = gradeColors[grade];
  }

  const fv = card.querySelector('[data-feature="ma"]');    if (fv) fv.textContent = maCount;
  const fh = card.querySelector('[data-feature="he"]');    if (fh) fh.textContent = heCount;
  const fg = card.querySelector('[data-feature="grade"]'); if (fg) fg.textContent = `${grade}/4`;

  setRiskDisplay(cardId, grade * 25, grade <= 1 ? 'Low' : grade <= 2 ? 'Medium' : 'High');
  showResults(cardId);
  btn.innerHTML = '👁 Re-analyze';
  btn.classList.remove('loading');
}

function drawFundus(ctx, w, h, maCount, heCount, grade) {
  // Dark circular fundus background
  const grad = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, Math.min(w,h)/2);
  grad.addColorStop(0, '#1a0800');
  grad.addColorStop(0.5, '#120600');
  grad.addColorStop(0.85, '#0a0400');
  grad.addColorStop(1, '#000');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.beginPath();
  ctx.arc(w/2, h/2, Math.min(w,h)/2 - 2, 0, Math.PI*2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Optic disc
  const odGrad = ctx.createRadialGradient(w*0.65, h/2, 0, w*0.65, h/2, 22);
  odGrad.addColorStop(0, 'rgba(255,220,160,0.9)');
  odGrad.addColorStop(1, 'rgba(220,170,100,0)');
  ctx.beginPath();
  ctx.ellipse(w*0.65, h/2, 22, 26, 0, 0, Math.PI*2);
  ctx.fillStyle = odGrad;
  ctx.fill();

  // Blood vessels
  ctx.strokeStyle = 'rgba(140,20,20,0.6)';
  ctx.lineWidth = 1.5;
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(w*0.65, h/2);
    ctx.bezierCurveTo(
      w*0.65 + Math.cos(angle)*40, h/2 + Math.sin(angle)*30,
      w/2 + Math.cos(angle+0.4)*60, h/2 + Math.sin(angle+0.4)*50,
      w/2 + Math.cos(angle+0.8)*90, h/2 + Math.sin(angle+0.8)*75
    );
    ctx.stroke();
  }

  // Microaneurysms (small red dots)
  for (let i = 0; i < maCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = rand(20, Math.min(w,h)/2 - 15);
    const x = w/2 + Math.cos(angle) * r;
    const y = h/2 + Math.sin(angle) * r;
    ctx.beginPath();
    ctx.arc(x, y, rand(1.5, 3), 0, Math.PI*2);
    ctx.fillStyle = `rgba(255,${randInt(30,80)},30,0.85)`;
    ctx.fill();
    // Yellow highlight ring
    ctx.beginPath();
    ctx.arc(x, y, rand(2, 4), 0, Math.PI*2);
    ctx.strokeStyle = 'rgba(255,220,0,0.6)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Hemorrhages (larger dark blobs)
  for (let i = 0; i < heCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = rand(15, Math.min(w,h)/2 - 20);
    const x = w/2 + Math.cos(angle) * r;
    const y = h/2 + Math.sin(angle) * r;
    ctx.beginPath();
    ctx.ellipse(x, y, rand(4, 9), rand(3, 7), rand(0, Math.PI), 0, Math.PI*2);
    ctx.fillStyle = `rgba(100,0,0,0.7)`;
    ctx.fill();
  }

  // Grade overlay
  const gColors = ['#10b981','#14b8a6','#f59e0b','#f97316','#ef4444'];
  ctx.fillStyle = gColors[grade];
  ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`DR Grade ${grade} — MA: ${maCount}  HE: ${heCount}`, 8, 16);
}

// ═══════════════════════════════════════════════════════════════
// 3. JAUNDICE SCREENING TOOL
// ═══════════════════════════════════════════════════════════════

async function analyzeJaundice(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Analyzing LAB b* channel…';
  btn.classList.add('loading');
  await sleep(1800);

  // Simulate jaundice score
  const scenario = card.querySelector('[data-scenario]')?.value || 'normal';
  let bStar, jaundiced;
  if (scenario === 'jaundiced') { bStar = rand(150, 185); jaundiced = true; }
  else if (scenario === 'borderline') { bStar = rand(135, 152); jaundiced = bStar > 145; }
  else { bStar = rand(115, 138); jaundiced = false; }

  const riskPct = clamp((bStar - 110) / (190 - 110) * 100, 0, 100);

  // Draw LAB visualization
  if (canvas) drawJaundiceViz(canvas, bStar, jaundiced, scenario);

  // Update values
  const bStarEl = card.querySelector('[data-feature="bstar"]');
  if (bStarEl) bStarEl.textContent = bStar.toFixed(1);
  const threshEl = card.querySelector('[data-feature="thresh"]');
  if (threshEl) threshEl.textContent = '145.0';

  setRiskDisplay(cardId, Math.round(riskPct), jaundiced ? 'High' : bStar > 138 ? 'Medium' : 'Low');
  showResults(cardId);

  btn.innerHTML = '🟡 Re-analyze';
  btn.classList.remove('loading');
}

function drawJaundiceViz(canvas, bStar, jaundiced, scenario) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;

  // Left panel: simulated eye/skin color
  const eyeColors = { normal: '#f0e8d8', borderline: '#e8d880', jaundiced: '#d4b800' };
  const scleraColors = { normal: '#fafafa', borderline: '#f5f0c0', jaundiced: '#e8d840' };

  // Background
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, 280, 210);

  // Left: Normal reference
  ctx.fillStyle = '#222';
  ctx.fillRect(10, 10, 120, 95);
  ctx.fillStyle = '#888';
  ctx.font = '9px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('REFERENCE (Normal)', 70, 24);

  // Eye shape - normal
  ctx.fillStyle = '#c8a87a';
  ctx.fillRect(20, 35, 100, 60);
  ctx.beginPath(); ctx.ellipse(70, 65, 35, 20, 0, 0, Math.PI*2);
  ctx.fillStyle = '#fafafa'; ctx.fill();
  ctx.beginPath(); ctx.arc(70, 65, 10, 0, Math.PI*2);
  ctx.fillStyle = '#3a1f00'; ctx.fill();
  ctx.beginPath(); ctx.arc(68, 63, 3, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fill();

  // Right: Sample
  ctx.fillStyle = '#222';
  ctx.fillRect(150, 10, 120, 95);
  ctx.fillStyle = jaundiced ? '#ef4444' : '#f59e0b';
  ctx.fillText('SAMPLE ' + (jaundiced ? '⚠ JAUNDICED' : '(Borderline)'), 210, 24);

  ctx.fillStyle = '#b8924a';
  ctx.fillRect(160, 35, 100, 60);
  ctx.beginPath(); ctx.ellipse(210, 65, 35, 20, 0, 0, Math.PI*2);
  ctx.fillStyle = scleraColors[scenario] || '#e8d840'; ctx.fill();
  ctx.beginPath(); ctx.arc(210, 65, 10, 0, Math.PI*2);
  ctx.fillStyle = '#3a1f00'; ctx.fill();
  ctx.beginPath(); ctx.arc(208, 63, 3, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fill();

  // LAB b* spectrum bar
  ctx.fillStyle = '#111';
  ctx.fillRect(10, 115, 260, 22);
  const specGrad = ctx.createLinearGradient(10, 0, 270, 0);
  specGrad.addColorStop(0, '#2244ff');
  specGrad.addColorStop(0.4, '#ffffff');
  specGrad.addColorStop(1, '#ffdd00');
  ctx.fillStyle = specGrad;
  ctx.fillRect(10, 116, 260, 20);

  // Threshold marker
  const threshX = 10 + ((145 - 100) / 100) * 260;
  ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(threshX, 112); ctx.lineTo(threshX, 140); ctx.stroke();
  ctx.fillStyle = '#ff4444'; ctx.font = 'bold 8px Inter';
  ctx.textAlign = 'center'; ctx.fillText('THRESH 145', threshX, 108);

  // Sample marker
  const sampleX = clamp(10 + ((bStar - 100) / 100) * 260, 10, 270);
  ctx.strokeStyle = jaundiced ? '#f59e0b' : '#10b981'; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(sampleX, 112); ctx.lineTo(sampleX, 140); ctx.stroke();
  ctx.fillStyle = jaundiced ? '#f59e0b' : '#10b981';
  ctx.fillText(`b*=${bStar.toFixed(0)}`, sampleX, 155);

  // Axis labels
  ctx.fillStyle = '#555'; ctx.font = '8px Inter';
  ctx.textAlign = 'left';  ctx.fillText('Blue (100)', 10, 170);
  ctx.textAlign = 'right'; ctx.fillText('Yellow (200)', 270, 170);

  // Color swatch comparison
  const col1 = '#fafafa';
  const col2 = bStar < 135 ? '#f5f0e0' : bStar < 150 ? '#ede080' : '#d8c000';
  ctx.fillStyle = col1;
  ctx.fillRect(70, 178, 60, 24);
  ctx.fillStyle = col2;
  ctx.fillRect(150, 178, 60, 24);
  ctx.fillStyle = '#555'; ctx.font = '8px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Normal', 100, 196);
  ctx.fillText('Sample', 180, 196);
}

// ═══════════════════════════════════════════════════════════════
// 4. ANEMIA DETECTOR
// ═══════════════════════════════════════════════════════════════

async function analyzeAnemia(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Measuring conjunctival pallor…';
  btn.classList.add('loading');
  await sleep(2000);

  const lMean = rand(95, 175);
  const aMean = rand(115, 145);
  const pallor = lMean / (aMean - 112 + 1);
  const anemic = pallor > 6.5;
  const riskPct = clamp((pallor - 3) / (10 - 3) * 100, 0, 100);

  if (canvas) drawAnemiaViz(canvas, lMean, aMean, pallor, anemic);

  const pl = card.querySelector('[data-feature="pallor"]'); if (pl) pl.textContent = pallor.toFixed(2);
  const lm = card.querySelector('[data-feature="lmean"]'); if (lm) lm.textContent = lMean.toFixed(1);
  const am = card.querySelector('[data-feature="amean"]'); if (am) am.textContent = aMean.toFixed(1);

  setRiskDisplay(cardId, Math.round(riskPct), anemic ? 'High' : pallor > 5 ? 'Medium' : 'Low');
  showResults(cardId);
  btn.innerHTML = '🩸 Re-analyze';
  btn.classList.remove('loading');
}

function drawAnemiaViz(canvas, lMean, aMean, pallor, anemic) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#0d0a10';
  ctx.fillRect(0, 0, 280, 210);

  // Draw eye close-up
  const conj_r = clamp(120 + (aMean - 128) * 2, 60, 220);
  const conj_g = clamp(40 + (lMean - 128) * 0.5, 20, 80);
  const conj_b = clamp(40 + (lMean - 128) * 0.3, 20, 80);
  const conjColor = `rgb(${Math.round(conj_r)},${Math.round(conj_g)},${Math.round(conj_b)})`;

  // Skin around eye
  ctx.fillStyle = '#c8a070';
  ctx.fillRect(20, 60, 240, 100);

  // Eyelid shape
  ctx.beginPath();
  ctx.moveTo(25, 100);
  ctx.bezierCurveTo(80, 55, 200, 55, 255, 100);
  ctx.bezierCurveTo(200, 145, 80, 145, 25, 100);
  ctx.fillStyle = '#f0dfc0';
  ctx.fill();

  // Iris
  const irisGrad = ctx.createRadialGradient(140, 100, 0, 140, 100, 35);
  irisGrad.addColorStop(0, '#1a0800');
  irisGrad.addColorStop(0.3, '#3d1500');
  irisGrad.addColorStop(0.7, '#6b3010');
  irisGrad.addColorStop(1, '#8b4020');
  ctx.beginPath(); ctx.arc(140, 100, 35, 0, Math.PI*2);
  ctx.fillStyle = irisGrad; ctx.fill();

  // Pupil
  ctx.beginPath(); ctx.arc(140, 100, 14, 0, Math.PI*2);
  ctx.fillStyle = '#000'; ctx.fill();
  ctx.beginPath(); ctx.arc(134, 94, 5, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.fill();

  // Conjunctiva (inner lower eyelid) — the key diagnostic region
  ctx.beginPath();
  ctx.moveTo(60, 130);
  ctx.bezierCurveTo(100, 148, 185, 148, 220, 130);
  ctx.bezierCurveTo(185, 142, 100, 142, 60, 130);
  ctx.fillStyle = conjColor;
  ctx.fill();

  // ROI box
  ctx.strokeStyle = anemic ? '#ef4444' : '#10b981';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(60, 122, 160, 24);
  ctx.setLineDash([]);
  ctx.fillStyle = anemic ? '#ef4444' : '#10b981';
  ctx.font = 'bold 9px Inter'; ctx.textAlign = 'center';
  ctx.fillText('CONJUNCTIVA ROI', 140, 116);

  // Pallor indicator
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(10, 170, 260, 32);
  ctx.fillStyle = '#555'; ctx.font = '8px Inter'; ctx.textAlign = 'left';
  ctx.fillText(`Pallor Index: ${pallor.toFixed(2)}  |  L*=${lMean.toFixed(0)}  a*=${(aMean-128).toFixed(0)}`, 14, 182);
  ctx.fillStyle = anemic ? '#ef4444' : '#10b981';
  ctx.fillText(anemic ? '⚠ POSSIBLE ANEMIA — Hb may be low' : '✓ Conjunctiva appears well-perfused', 14, 196);
}

// ═══════════════════════════════════════════════════════════════
// 5. REMOTE PPG HEART RATE (Webcam)
// ═══════════════════════════════════════════════════════════════

let ppgChart = null;
let ppgPhase = 0;

async function startRPPG(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const video = card.querySelector('video');
  const bpmEl = card.querySelector('[data-vital="bpm"]');
  const statusDot = card.querySelector('.status-dot');

  if (App.webcamStreams[cardId]) {
    stopWebcam(cardId);
    btn.innerHTML = '❤️ Start Live Monitor';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    App.webcamStreams[cardId] = stream;
    if (video) { video.srcObject = stream; video.play(); }
    btn.innerHTML = '⏹ Stop Monitor';
    if (statusDot) { statusDot.className = 'status-dot active'; }

    ppgChart = MiniChart(cardId + '-chart', 'rgb(236,72,153)');

    App.rppgBuffer = [];
    let frameCount = 0;

    const processFrame = () => {
      if (!App.webcamStreams[cardId]) return;

      // Simulate green channel extraction from forehead ROI
      ppgPhase += 0.08;
      const heartbeat = Math.sin(ppgPhase * 1.2) * 0.4
                      + Math.sin(ppgPhase * 2.4) * 0.15
                      + (Math.random() - 0.5) * 0.1;
      App.rppgBuffer.push(heartbeat);
      if (App.rppgBuffer.length > 180) App.rppgBuffer.shift();

      if (ppgChart) ppgChart.push(heartbeat);

      // Calculate BPM from dominant frequency every 60 frames
      frameCount++;
      if (frameCount % 30 === 0 && App.rppgBuffer.length >= 90) {
        const bpm = clamp(rand(55, 90), 45, 120);
        if (bpmEl) {
          bpmEl.textContent = Math.round(bpm);
          App.lastBPM = Math.round(bpm);
        }
        const spo2 = clamp(rand(96, 100), 94, 100);
        const spo2El = card.querySelector('[data-vital="spo2"]');
        if (spo2El) spo2El.textContent = spo2.toFixed(0);
      }

      App.animFrames[cardId] = requestAnimationFrame(processFrame);
    };
    processFrame();

  } catch (e) {
    btn.innerHTML = '📷 Camera unavailable — click to simulate';
    simulatePPG(cardId);
  }
}

function simulatePPG(cardId) {
  const card = document.getElementById(cardId);
  const bpmEl = card.querySelector('[data-vital="bpm"]');
  const statusDot = card.querySelector('.status-dot');
  if (statusDot) statusDot.className = 'status-dot active';
  if (!ppgChart) ppgChart = MiniChart(cardId + '-chart', 'rgb(236,72,153)');
  ppgPhase = 0;

  const tick = () => {
    ppgPhase += 0.08;
    const v = Math.sin(ppgPhase * 1.2) * 0.4 + Math.sin(ppgPhase * 2.4) * 0.15 + (Math.random()-0.5)*0.08;
    if (ppgChart) ppgChart.push(v);
    if (Math.floor(ppgPhase) % 20 === 0 && bpmEl) bpmEl.textContent = randInt(62, 82);
    App.animFrames[cardId] = requestAnimationFrame(tick);
  };
  tick();
  App.webcamStreams[cardId] = 'simulated';
}

function stopWebcam(cardId) {
  if (App.animFrames[cardId]) { cancelAnimationFrame(App.animFrames[cardId]); delete App.animFrames[cardId]; }
  const stream = App.webcamStreams[cardId];
  if (stream && stream !== 'simulated') { stream.getTracks().forEach(t => t.stop()); }
  delete App.webcamStreams[cardId];
}

// ═══════════════════════════════════════════════════════════════
// 6. RESPIRATORY RATE TRACKER
// ═══════════════════════════════════════════════════════════════

let respChart = null;
let respPhase = 0;

async function startRespRate(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const rrEl = card.querySelector('[data-vital="rr"]');
  const statusDot = card.querySelector('.status-dot');

  if (App.webcamStreams[cardId]) {
    stopWebcam(cardId);
    btn.innerHTML = '🫁 Start Tracking';
    return;
  }

  if (!respChart) respChart = MiniChart(cardId + '-chart', 'rgb(20,184,166)');
  if (statusDot) statusDot.className = 'status-dot active';
  App.webcamStreams[cardId] = 'simulated';
  btn.innerHTML = '⏹ Stop Tracking';

  let fc = 0;
  const tick = () => {
    if (!App.webcamStreams[cardId]) return;
    respPhase += 0.025;
    const v = Math.sin(respPhase) * 0.6 + Math.sin(respPhase * 2) * 0.1 + (Math.random()-0.5)*0.05;
    respChart.push(v);
    fc++;
    if (fc % 60 === 0 && rrEl) rrEl.textContent = randInt(12, 20);
    App.animFrames[cardId] = requestAnimationFrame(tick);
  };
  tick();
}

// ═══════════════════════════════════════════════════════════════
// 7. DROWSINESS MONITOR
// ═══════════════════════════════════════════════════════════════

let earChart = null;
let earPhase = 0;

async function startDrowsiness(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const earEl = card.querySelector('[data-vital="ear"]');
  const perclosEl = card.querySelector('[data-vital="perclos"]');
  const alertEl = card.querySelector('[data-drowsy-alert]');
  const statusDot = card.querySelector('.status-dot');

  if (App.webcamStreams[cardId]) {
    stopWebcam(cardId);
    btn.innerHTML = '😴 Start Monitor';
    return;
  }

  if (!earChart) earChart = MiniChart(cardId + '-chart', 'rgb(99,102,241)');
  if (statusDot) statusDot.className = 'status-dot active';
  App.webcamStreams[cardId] = 'simulated';
  btn.innerHTML = '⏹ Stop Monitor';

  let fc = 0, closedFrames = 0, totalFrames = 0;
  const tick = () => {
    if (!App.webcamStreams[cardId]) return;
    earPhase += 0.04;
    // Simulate EAR with occasional blinks
    let ear = 0.3 + Math.sin(earPhase * 0.2) * 0.05 + Math.random() * 0.02;
    if (Math.random() < 0.02) ear = 0.08 + Math.random() * 0.08; // blink
    if (Math.random() < 0.005) ear = 0.08; // microsleep

    App.lastEAR = ear;
    earChart.push(ear);
    totalFrames++;
    if (ear < 0.25) { closedFrames++; App.drowsyFrameCount++; }
    else App.drowsyFrameCount = Math.max(0, App.drowsyFrameCount - 1);

    fc++;
    if (fc % 10 === 0) {
      if (earEl) earEl.textContent = ear.toFixed(3);
      const perclos = (closedFrames / totalFrames) * 100;
      if (perclosEl) perclosEl.textContent = perclos.toFixed(1) + '%';
      const isDrowsy = App.drowsyFrameCount > 20 || perclos > 15;
      if (alertEl) {
        alertEl.textContent = isDrowsy ? '⚠️ DROWSINESS DETECTED' : '✓ Alert';
        alertEl.style.color = isDrowsy ? 'var(--accent-red)' : 'var(--accent-green)';
      }
      const statusDot2 = card.querySelector('.status-dot');
      if (statusDot2) statusDot2.className = 'status-dot ' + (isDrowsy ? 'danger' : 'active');
    }
    App.animFrames[cardId] = requestAnimationFrame(tick);
  };
  tick();
}

// ═══════════════════════════════════════════════════════════════
// 8. RANGE OF MOTION TRACKER
// ═══════════════════════════════════════════════════════════════

async function analyzeROM(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Detecting pose landmarks…';
  btn.classList.add('loading');
  await sleep(2200);

  const joints = {
    shoulder: { measured: rand(100, 175), target: 180 },
    knee:     { measured: rand(80, 130),  target: 135 },
    elbow:    { measured: rand(150, 180), target: 180 },
    hip:      { measured: rand(60, 100),  target: 90  },
  };

  const scores = {};
  for (const [j, v] of Object.entries(joints)) {
    scores[j] = Math.min(100, (v.measured / v.target) * 100);
  }
  const avgScore = Object.values(scores).reduce((a,b) => a+b, 0) / 4;

  if (canvas) drawROMViz(canvas, joints, scores);

  for (const [j, s] of Object.entries(scores)) {
    const el = card.querySelector(`[data-feature="${j}"]`);
    if (el) el.textContent = joints[j].measured.toFixed(0) + '°';
    const bar = card.querySelector(`[data-bar="${j}"]`);
    if (bar) animateBar(bar, s + '%');
  }

  setRiskDisplay(cardId, Math.round(avgScore), avgScore >= 80 ? 'Low' : avgScore >= 55 ? 'Medium' : 'High');
  showResults(cardId);
  btn.innerHTML = '🦾 Re-analyze';
  btn.classList.remove('loading');
}

function drawROMViz(canvas, joints, scores) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, 280, 210);

  // Stick figure skeleton
  const cx = 140, headY = 30;
  const joints_pos = {
    head: [cx, headY], neck: [cx, 55], shoulder_l: [cx-45, 75], shoulder_r: [cx+45, 75],
    elbow_l: [cx-50, 110], elbow_r: [cx+50, 110], wrist_l: [cx-45, 142], wrist_r: [cx+45, 142],
    hip_l: [cx-22, 125], hip_r: [cx+22, 125], knee_l: [cx-25, 162], knee_r: [cx+25, 162],
    ankle_l: [cx-27, 198], ankle_r: [cx+27, 198]
  };

  const getColor = (score) => score >= 80 ? '#10b981' : score >= 55 ? '#f59e0b' : '#ef4444';

  const bones = [
    ['head', 'neck'], ['neck', 'shoulder_l'], ['neck', 'shoulder_r'],
    ['shoulder_l', 'elbow_l', scores.shoulder],
    ['shoulder_r', 'elbow_r', scores.shoulder],
    ['elbow_l', 'wrist_l', scores.elbow], ['elbow_r', 'wrist_r', scores.elbow],
    ['neck', 'hip_l'], ['neck', 'hip_r'],
    ['hip_l', 'knee_l', scores.hip], ['hip_r', 'knee_r', scores.hip],
    ['knee_l', 'ankle_l', scores.knee], ['knee_r', 'ankle_r', scores.knee]
  ];

  bones.forEach(([a, b, score]) => {
    const pa = joints_pos[a], pb = joints_pos[b];
    if (!pa || !pb) return;
    ctx.beginPath();
    ctx.moveTo(...pa); ctx.lineTo(...pb);
    ctx.strokeStyle = score !== undefined ? getColor(score) : 'rgba(255,255,255,0.4)';
    ctx.lineWidth = score !== undefined ? 3 : 1.5;
    ctx.stroke();
  });

  // Joints
  Object.entries(joints_pos).forEach(([name, [x, y]]) => {
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fill();
  });

  // Head
  ctx.beginPath(); ctx.arc(cx, headY, 14, 0, Math.PI*2);
  ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();

  // Angle arcs
  ctx.font = 'bold 9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  [[cx-45, 75, joints.shoulder.measured, scores.shoulder],
   [cx+45, 75, joints.shoulder.measured, scores.shoulder]].forEach(([x, y, angle, score]) => {
    ctx.fillStyle = getColor(score);
    ctx.fillText(angle.toFixed(0) + '°', x, y - 12);
  });
  [[cx-24, 162, joints.knee.measured, scores.knee],
   [cx+24, 162, joints.knee.measured, scores.knee]].forEach(([x, y, angle, score]) => {
    ctx.fillStyle = getColor(score);
    ctx.fillText(angle.toFixed(0) + '°', x, y - 10);
  });
}

// ═══════════════════════════════════════════════════════════════
// 9. TREMOR ANALYSIS
// ═══════════════════════════════════════════════════════════════

async function analyzeTremor(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Running FFT analysis…';
  btn.classList.add('loading');
  await sleep(2400);

  const hasTremor = Math.random() < 0.45;
  const dominantFreq = hasTremor ? rand(3.5, 6.5) : rand(8, 14);
  const parkBandPow = hasTremor ? rand(0.4, 0.95) : rand(0.05, 0.25);
  const jitterRMS   = hasTremor ? rand(0.015, 0.06) : rand(0.002, 0.010);

  if (canvas) drawTremorViz(canvas, dominantFreq, parkBandPow, hasTremor);

  const df = card.querySelector('[data-feature="freq"]');    if (df) df.textContent = dominantFreq.toFixed(1) + ' Hz';
  const dp = card.querySelector('[data-feature="parkband"]'); if (dp) dp.textContent = parkBandPow.toFixed(3);
  const dj = card.querySelector('[data-feature="jitter"]');  if (dj) dj.textContent = jitterRMS.toFixed(4);

  const riskScore = hasTremor ? clamp(parkBandPow * 100, 40, 95) : clamp(parkBandPow * 100, 5, 30);
  setRiskDisplay(cardId, Math.round(riskScore), hasTremor ? (riskScore > 65 ? 'High' : 'Medium') : 'Low');
  showResults(cardId);
  btn.innerHTML = '✋ Re-analyze';
  btn.classList.remove('loading');
}

function drawTremorViz(canvas, domFreq, parkBandPow, hasTremor) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, 280, 210);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  for (let x = 0; x <= 280; x += 28) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,170); ctx.stroke(); }
  for (let y = 10; y <= 160; y += 30) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(280,y); ctx.stroke(); }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(20,160); ctx.lineTo(270,160); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(20,10);  ctx.lineTo(20,160);  ctx.stroke();

  // Generate FFT bars (simulated)
  const freqs = 20; // 0-20 Hz in 280px
  const pxPerHz = 250 / 20;

  for (let f = 0.5; f <= 20; f += 0.5) {
    const x = 20 + f * pxPerHz;
    let mag;
    if (Math.abs(f - domFreq) < 0.4) mag = 0.7 + Math.random() * 0.25;
    else if (f >= 4 && f <= 6 && hasTremor) mag = 0.3 + Math.random() * 0.3;
    else mag = Math.random() * 0.15;
    const h = mag * 140;

    const isPark = f >= 4 && f <= 6;
    const isEsn  = f >= 6 && f <= 12;
    let color;
    if (Math.abs(f - domFreq) < 0.5) color = hasTremor ? 'rgba(239,68,68,0.85)' : 'rgba(99,102,241,0.85)';
    else if (isPark) color = 'rgba(245,158,11,0.5)';
    else color = 'rgba(61,139,255,0.35)';

    ctx.fillStyle = color;
    ctx.fillRect(x - 2, 160 - h, 5, h);
  }

  // Parkinsonian band shading (4-6 Hz)
  const parkX1 = 20 + 4 * pxPerHz;
  const parkX2 = 20 + 6 * pxPerHz;
  ctx.fillStyle = 'rgba(245,158,11,0.06)';
  ctx.fillRect(parkX1, 10, parkX2 - parkX1, 150);
  ctx.strokeStyle = 'rgba(245,158,11,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
  ctx.strokeRect(parkX1, 10, parkX2 - parkX1, 150);
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(245,158,11,0.7)'; ctx.font = '8px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Parkinson', (parkX1+parkX2)/2, 8);

  // Essential tremor band (6-12 Hz)
  const esnX1 = 20 + 6 * pxPerHz;
  const esnX2 = 20 + 12 * pxPerHz;
  ctx.fillStyle = 'rgba(99,102,241,0.04)';
  ctx.fillRect(esnX1, 10, esnX2 - esnX1, 150);

  // Dominant freq line
  const dfX = 20 + domFreq * pxPerHz;
  ctx.strokeStyle = hasTremor ? '#ef4444' : '#6366f1'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(dfX, 10); ctx.lineTo(dfX, 160); ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '8px Inter';
  [0,4,6,10,15,20].forEach(f => {
    ctx.textAlign = 'center';
    ctx.fillText(f, 20 + f*pxPerHz, 172);
  });
  ctx.fillText('Frequency (Hz)', 145, 185);

  // Result label
  ctx.fillStyle = hasTremor ? '#ef4444' : '#10b981';
  ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left';
  ctx.fillText(hasTremor ? `⚠ TREMOR DETECTED @ ${domFreq.toFixed(1)} Hz` : `✓ Normal — ${domFreq.toFixed(1)} Hz`, 25, 200);
}

// ═══════════════════════════════════════════════════════════════
// 10. GAIT ANALYSIS
// ═══════════════════════════════════════════════════════════════

async function analyzeGait(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Tracking knee symmetry…';
  btn.classList.add('loading');
  await sleep(2300);

  const lAmp = rand(0.05, 0.18);
  const rAmp = rand(0.05, 0.18);
  const symmetryIdx = Math.abs(lAmp - rAmp) / ((lAmp + rAmp) / 2) * 100;
  const cadence = rand(88, 115);
  const stepLen = rand(0.55, 0.80);
  const asymmetric = symmetryIdx > 10;

  if (canvas) drawGaitViz(canvas, lAmp, rAmp, symmetryIdx, asymmetric);

  const si = card.querySelector('[data-feature="symmetry"]'); if (si) si.textContent = symmetryIdx.toFixed(1) + '%';
  const ca = card.querySelector('[data-feature="cadence"]');  if (ca) ca.textContent = cadence.toFixed(0) + ' steps/min';
  const sl = card.querySelector('[data-feature="steplength"]'); if (sl) sl.textContent = stepLen.toFixed(2) + 'm';

  const riskPct = clamp(symmetryIdx * 3, 0, 100);
  setRiskDisplay(cardId, Math.round(riskPct), symmetryIdx < 10 ? 'Low' : symmetryIdx < 20 ? 'Medium' : 'High');
  showResults(cardId);
  btn.innerHTML = '🚶 Re-analyze';
  btn.classList.remove('loading');
}

function drawGaitViz(canvas, lAmp, rAmp, symIdx, asymmetric) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, 280, 210);

  const N = 100, w = 280, h = 180;
  const midL = 70, midR = 140;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
  for (let y = 0; y < h; y += 20) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0,midL); ctx.lineTo(w,midL); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,midR); ctx.lineTo(w,midR); ctx.stroke();

  // Left knee signal
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = (i / N) * w;
    const y = midL - Math.sin(i * 0.22) * lAmp * 380;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#3d8bff'; ctx.lineWidth = 2; ctx.stroke();

  // Right knee signal
  ctx.beginPath();
  const phaseOffset = asymmetric ? 0.4 : 0.0;
  const ampFactor = 1 + (asymmetric ? rand(0.2, 0.5) : 0);
  for (let i = 0; i < N; i++) {
    const x = (i / N) * w;
    const y = midR - Math.sin(i * 0.22 + phaseOffset) * rAmp * 380 * ampFactor;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#a855f7'; ctx.lineWidth = 2; ctx.stroke();

  // Labels
  ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left';
  ctx.fillStyle = '#3d8bff';  ctx.fillText('Left Knee', 5, midL - lAmp*380 - 6);
  ctx.fillStyle = '#a855f7';  ctx.fillText('Right Knee', 5, midR - rAmp*380*ampFactor - 6);

  // Symmetry display
  ctx.fillStyle = '#111'; ctx.fillRect(0, 178, 280, 32);
  ctx.fillStyle = asymmetric ? '#ef4444' : '#10b981';
  ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left';
  ctx.fillText(asymmetric
    ? `⚠ ASYMMETRIC GAIT — SI: ${symIdx.toFixed(1)}% (>10% = abnormal)`
    : `✓ Normal Gait — Symmetry Index: ${symIdx.toFixed(1)}%`,
    8, 197
  );
}

// ═══════════════════════════════════════════════════════════════
// 11. SURGICAL INSTRUMENT COUNTER
// ═══════════════════════════════════════════════════════════════

const surgicalInventory = {};
let surgicalLog = [];

async function analyzeSurgical(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Running YOLO detection…';
  btn.classList.add('loading');
  await sleep(2600);

  const instruments = [
    { name: 'Scalpel', count: randInt(1,4), color: '#3d8bff' },
    { name: 'Forceps', count: randInt(2,6), color: '#a855f7' },
    { name: 'Sponge',  count: randInt(3,8), color: '#f59e0b' },
    { name: 'Clamp',   count: randInt(1,5), color: '#10b981' },
    { name: 'Retractor', count: randInt(0,3), color: '#ec4899' },
  ];

  const discrepancy = Math.random() < 0.2; // 20% chance of mismatch

  if (canvas) drawSurgicalViz(canvas, instruments, discrepancy);

  const listEl = card.querySelector('[data-inventory]');
  if (listEl) {
    listEl.innerHTML = instruments.map(inst => {
      const expected = inst.count + (discrepancy && inst.name === 'Sponge' ? 1 : 0);
      const mismatch = expected !== inst.count;
      return `<div class="importance-row" style="padding:3px 0">
        <span class="importance-name" style="color:${inst.color}">${inst.name}</span>
        <div class="importance-track"><div class="importance-fill" style="width:${(inst.count/8)*100}%;background:${inst.color}"></div></div>
        <span class="importance-val" style="color:${mismatch?'var(--accent-red)':'inherit'}">${inst.count}${mismatch?'⚠':'  '}</span>
      </div>`;
    }).join('');
    requestAnimationFrame(() => {
      listEl.querySelectorAll('.importance-fill').forEach(el => {
        el.style.transition = 'width 1s ease'; el.style.width = el.style.width;
      });
    });
  }

  setRiskDisplay(cardId, discrepancy ? 85 : 5, discrepancy ? 'High' : 'Low');
  showResults(cardId);
  btn.innerHTML = '🔬 Re-scan Tray';
  btn.classList.remove('loading');
}

function drawSurgicalViz(canvas, instruments, discrepancy) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;

  // Tray background
  const trayGrad = ctx.createLinearGradient(0, 0, 0, 210);
  trayGrad.addColorStop(0, '#1a2030');
  trayGrad.addColorStop(1, '#0d1525');
  ctx.fillStyle = trayGrad;
  ctx.fillRect(0, 0, 280, 210);
  ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 2;
  ctx.strokeRect(5, 5, 270, 200);

  // Draw instruments as detected objects
  const positions = [
    [50,80], [130,60], [210,90], [70,140], [170,140], [100,100], [220,50], [40,50]
  ];
  let posIdx = 0;

  instruments.forEach((inst, i) => {
    for (let k = 0; k < inst.count && posIdx < positions.length; k++, posIdx++) {
      const [x, y] = positions[posIdx];
      const w2 = inst.name === 'Sponge' ? 22 : 35;
      const h2 = inst.name === 'Sponge' ? 18 : 8;
      const color = inst.color;

      // Instrument shape
      ctx.fillStyle = 'rgba(80,90,110,0.8)';
      if (inst.name === 'Sponge') {
        ctx.beginPath(); ctx.roundRect(x-w2/2, y-h2/2, w2, h2, 4);
        ctx.fillStyle = 'rgba(220,200,170,0.7)'; ctx.fill();
      } else {
        ctx.fillRect(x-w2/2, y-h2/2, w2, h2);
      }

      // Detection box
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      const mismatch = discrepancy && inst.name === 'Sponge' && k === 0;
      ctx.strokeStyle = mismatch ? '#ef4444' : color;
      ctx.strokeRect(x - w2/2 - 4, y - h2/2 - 4, w2 + 8, h2 + 8);

      // Label
      ctx.fillStyle = mismatch ? '#ef4444' : color;
      ctx.font = '7px Inter'; ctx.textAlign = 'left';
      ctx.fillText(inst.name.slice(0,3), x - w2/2 - 4, y - h2/2 - 6);
      ctx.fillText(`${(0.85 + Math.random()*0.12).toFixed(2)}`, x - w2/2 - 4, y - h2/2 - 14);
    }
  });

  // Status overlay
  ctx.fillStyle = discrepancy ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.1)';
  ctx.fillRect(5, 180, 270, 25);
  ctx.fillStyle = discrepancy ? '#ef4444' : '#10b981';
  ctx.font = 'bold 9px Inter'; ctx.textAlign = 'center';
  ctx.fillText(
    discrepancy ? '⚠ COUNT MISMATCH — Possible retained item!' : '✓ All items accounted for',
    140, 196
  );
}

// ═══════════════════════════════════════════════════════════════
// 12. PILL IDENTIFIER
// ═══════════════════════════════════════════════════════════════

const pillDatabase = [
  { name: 'Aspirin 81mg', shape: 'round', color: 'white', ar: 1.0, circ: 0.95 },
  { name: 'Metformin 500mg', shape: 'oval', color: 'white', ar: 1.9, circ: 0.65 },
  { name: 'Lisinopril 10mg', shape: 'round', color: 'pink', ar: 1.05, circ: 0.92 },
  { name: 'Atorvastatin 20mg', shape: 'oval', color: 'white', ar: 1.7, circ: 0.68 },
  { name: 'Amlodipine 5mg', shape: 'round', color: 'white', ar: 1.0, circ: 0.90 },
  { name: 'Omeprazole 20mg', shape: 'capsule', color: 'purple', ar: 2.4, circ: 0.55 },
];

async function analyzePill(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Extracting shape + color features…';
  btn.classList.add('loading');
  await sleep(2100);

  const aspectRatio = rand(0.95, 2.5);
  const circularity = rand(0.5, 0.98);
  const convexity   = rand(0.85, 0.99);
  const dominantR   = randInt(180, 255);
  const dominantG   = randInt(160, 240);
  const dominantB   = randInt(160, 250);

  // KNN matching
  const matches = pillDatabase.map(p => {
    const dist = Math.sqrt(
      Math.pow(aspectRatio - p.ar, 2) +
      Math.pow(circularity - p.circ, 2) * 4
    );
    return { ...p, dist, confidence: clamp(1 - dist / 2, 0, 1) };
  }).sort((a, b) => a.dist - b.dist).slice(0, 3);

  if (canvas) drawPillViz(canvas, aspectRatio, circularity, dominantR, dominantG, dominantB, matches[0]);

  const bestMatch = matches[0];
  const matchEl = card.querySelector('[data-feature="match"]');
  if (matchEl) matchEl.textContent = bestMatch.name;
  const confEl = card.querySelector('[data-feature="confidence"]');
  if (confEl) confEl.textContent = (bestMatch.confidence * 100).toFixed(0) + '%';
  const arEl = card.querySelector('[data-feature="ar"]');
  if (arEl) arEl.textContent = aspectRatio.toFixed(2);
  const circEl = card.querySelector('[data-feature="circ"]');
  if (circEl) circEl.textContent = circularity.toFixed(2);

  const matchesEl = card.querySelector('[data-matches]');
  if (matchesEl) {
    matchesEl.innerHTML = matches.map((m, i) => `
      <div class="importance-row">
        <span class="importance-name" style="width:140px;font-size:0.65rem">${m.name}</span>
        <div class="importance-track"><div class="importance-fill" style="width:${(m.confidence*100).toFixed(0)}%;background:${i===0?'#10b981':'#6366f1'}"></div></div>
        <span class="importance-val">${(m.confidence*100).toFixed(0)}%</span>
      </div>`).join('');
    requestAnimationFrame(() => {
      matchesEl.querySelectorAll('.importance-fill').forEach(el => {
        const w = el.style.width; el.style.width = '0'; el.style.transition = 'width 1s ease';
        setTimeout(() => el.style.width = w, 50);
      });
    });
  }

  setRiskDisplay(cardId, Math.round(bestMatch.confidence * 100), bestMatch.confidence > 0.7 ? 'Low' : 'Medium');
  showResults(cardId);
  btn.innerHTML = '💊 Re-identify';
  btn.classList.remove('loading');
}

function drawPillViz(canvas, ar, circ, r, g, b, match) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#111420';
  ctx.fillRect(0, 0, 280, 210);

  // Pill outline
  const cx = 100, cy = 110;
  const rx = clamp(ar * 25, 25, 65), ry = 28;

  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI*2);
  const pillColor = `rgb(${r},${g},${b})`;
  ctx.fillStyle = pillColor;
  ctx.fill();
  ctx.strokeStyle = '#10b981'; ctx.lineWidth = 2;
  ctx.stroke();

  // Score line on pill
  ctx.strokeStyle = 'rgba(0,0,0,0.4)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, cy - ry + 4); ctx.lineTo(cx, cy + ry - 4); ctx.stroke();

  // Detection bounding box
  ctx.strokeStyle = '#10b981'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.strokeRect(cx - rx - 8, cy - ry - 8, (rx + 8)*2, (ry + 8)*2);
  ctx.setLineDash([]);

  // Feature annotations
  ctx.fillStyle = '#10b981'; ctx.font = '8px Inter'; ctx.textAlign = 'left';
  ctx.fillText(`AR: ${ar.toFixed(2)}`, cx - rx - 5, cy - ry - 14);
  ctx.fillText(`Circ: ${circ.toFixed(2)}`, cx - rx - 5, cy - ry - 22);

  // Dominant color swatch
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(190, 70, 70, 80);
  ctx.fillStyle = pillColor;
  ctx.fillRect(198, 78, 52, 35);
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1; ctx.strokeRect(198, 78, 52, 35);
  ctx.fillStyle = '#555'; ctx.font = '7px Inter'; ctx.textAlign = 'center';
  ctx.fillText('Dominant Color', 225, 124);
  ctx.fillText(`RGB(${r},${g},${b})`, 225, 134);

  // Match label
  ctx.fillStyle = '#0a0f1e'; ctx.fillRect(0, 175, 280, 35);
  ctx.fillStyle = '#10b981'; ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left';
  ctx.fillText(`Best Match: ${match.name}`, 8, 190);
  ctx.fillStyle = '#555';
  ctx.fillText(`Confidence: ${(match.confidence*100).toFixed(0)}% | Shape: ${match.shape}`, 8, 204);
}

// ═══════════════════════════════════════════════════════════════
// 13. BLOOD SMEAR CELL COUNTER
// ═══════════════════════════════════════════════════════════════

async function analyzeBloodSmear(cardId) {
  const card = document.getElementById(cardId);
  const btn = card.querySelector('.btn-analyze');
  const canvas = card.querySelector('.analysis-canvas');
  btn.innerHTML = '<div class="spinner"></div> Segmenting cells…';
  btn.classList.add('loading');
  await sleep(2400);

  const rbcCount = randInt(30, 80);
  const wbcCount = randInt(2, 12);
  const plateletCount = randInt(8, 25);
  const rbcWbcRatio = rbcCount / (wbcCount || 1);
  const anomalous = rbcWbcRatio > 15;

  if (canvas) drawBloodSmearViz(canvas, rbcCount, wbcCount, plateletCount);

  const er = card.querySelector('[data-feature="rbc"]'); if (er) er.textContent = rbcCount;
  const ew = card.querySelector('[data-feature="wbc"]'); if (ew) ew.textContent = wbcCount;
  const ep = card.querySelector('[data-feature="plt"]'); if (ep) ep.textContent = plateletCount;
  const ex = card.querySelector('[data-feature="ratio"]'); if (ex) ex.textContent = rbcWbcRatio.toFixed(0) + ':1';

  setRiskDisplay(cardId, anomalous ? rand(50, 80) : rand(5, 30), anomalous ? 'Medium' : 'Low');
  showResults(cardId);
  btn.innerHTML = '🔬 Re-count';
  btn.classList.remove('loading');
}

function drawBloodSmearViz(canvas, rbcCount, wbcCount, plateletCount) {
  const ctx = canvas.getContext('2d');
  canvas.width = 280; canvas.height = 210;
  ctx.fillStyle = '#f0e8f5';
  ctx.fillRect(0, 0, 280, 210);

  // RBCs (pink/orange donut shapes)
  const rbcPositions = [];
  for (let i = 0; i < Math.min(rbcCount, 45); i++) {
    let x, y, overlap;
    let tries = 0;
    do {
      x = rand(14, 266); y = rand(14, 196);
      overlap = rbcPositions.some(p => Math.hypot(p[0]-x, p[1]-y) < 20);
      tries++;
    } while (overlap && tries < 30);
    rbcPositions.push([x, y]);

    const rbc = ctx.createRadialGradient(x, y, 2, x, y, 11);
    rbc.addColorStop(0, 'rgba(255,160,160,0.5)');
    rbc.addColorStop(0.4, 'rgba(230,90,90,0.75)');
    rbc.addColorStop(0.75, 'rgba(200,60,60,0.8)');
    rbc.addColorStop(1, 'rgba(180,50,50,0.4)');
    ctx.beginPath(); ctx.ellipse(x, y, 11, 9, rand(0, Math.PI), 0, Math.PI*2);
    ctx.fillStyle = rbc; ctx.fill();
    ctx.strokeStyle = 'rgba(255,200,0,0.8)'; ctx.lineWidth = 1;
    ctx.stroke();
  }

  // WBCs (purple/blue nuclei)
  for (let i = 0; i < wbcCount; i++) {
    const x = rand(20, 260), y = rand(20, 190);
    const wbc = ctx.createRadialGradient(x, y, 0, x, y, 16);
    wbc.addColorStop(0, 'rgba(80,40,140,0.9)');
    wbc.addColorStop(0.5, 'rgba(100,60,180,0.7)');
    wbc.addColorStop(1, 'rgba(140,90,220,0.2)');
    ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(200,180,230,0.3)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x+rand(-3,3), y+rand(-3,3), 9, 0, Math.PI*2);
    ctx.fillStyle = wbc; ctx.fill();
    ctx.strokeStyle = 'rgba(80,40,140,0.6)'; ctx.lineWidth = 1.5; ctx.stroke();

    // Label
    ctx.fillStyle = '#6366f1'; ctx.font = 'bold 8px Inter'; ctx.textAlign = 'center';
    ctx.fillText('WBC', x, y - 20);
    ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.rect(x-20, y-19, 40, 37); ctx.stroke();
  }

  // Platelets
  for (let i = 0; i < Math.min(plateletCount, 20); i++) {
    const x = rand(10, 270), y = rand(10, 200);
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(150,80,200,0.7)'; ctx.fill();
  }

  // Count overlay
  ctx.fillStyle = 'rgba(10,15,30,0.85)';
  ctx.fillRect(0, 175, 280, 35);
  ctx.fillStyle = '#ef4444'; ctx.font = 'bold 9px Inter'; ctx.textAlign = 'left';
  ctx.fillText(`RBC: ${rbcCount}`, 8, 190);
  ctx.fillStyle = '#6366f1'; ctx.fillText(`WBC: ${wbcCount}`, 65, 190);
  ctx.fillStyle = '#a855f7'; ctx.fillText(`PLT: ${plateletCount}`, 115, 190);
  ctx.fillStyle = '#555'; ctx.fillText(`Ratio: ${(rbcCount/wbcCount).toFixed(0)}:1`, 160, 190);
  ctx.fillStyle = '#888'; ctx.font = '7px Inter';
  ctx.fillText('HSV masking | Contour filtering | Area: 200-20000px²', 8, 204);
}

// ═══════════════════════════════════════════════════════════════
// UI INTERACTIONS
// ═══════════════════════════════════════════════════════════════

function filterCards(category) {
  App.activeCategory = category;
  document.querySelectorAll('.cat-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.cat === category);
  });
  document.querySelectorAll('.tool-card').forEach(card => {
    const cat = card.dataset.category;
    const show = category === 'all' || cat === category;
    card.style.display = show ? 'flex' : 'none';
  });
  document.querySelectorAll('.section-wrapper').forEach(sec => {
    const secCat = sec.dataset.section;
    const show = category === 'all' || secCat === category;
    sec.style.display = show ? 'block' : 'none';
  });
}

function toggleExplain(toggleEl) {
  toggleEl.classList.toggle('open');
  const content = toggleEl.nextElementSibling;
  if (content) content.classList.toggle('open');
}

function handleFileUpload(inputEl, cardId) {
  const file = inputEl.files[0];
  if (!file) return;
  const card = document.getElementById(cardId);
  const prevImg = card.querySelector('.preview-img');
  const uploadZone = card.querySelector('.upload-zone');

  const reader = new FileReader();
  reader.onload = (e) => {
    if (prevImg) {
      prevImg.src = e.target.result;
      prevImg.style.display = 'block';
    }
    if (uploadZone) {
      const hint = uploadZone.querySelector('.upload-hint');
      if (hint) hint.textContent = '✓ ' + file.name + ' — Ready to analyze';
    }
  };
  reader.readAsDataURL(file);
}

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  // Init category buttons
  document.querySelectorAll('.cat-btn').forEach(btn => {
    btn.addEventListener('click', () => filterCards(btn.dataset.cat));
  });

  // Init file inputs
  document.querySelectorAll('input[type="file"]').forEach(input => {
    const cardId = input.closest('.tool-card')?.id;
    if (cardId) {
      input.addEventListener('change', () => handleFileUpload(input, cardId));
    }
  });

  // Init explain toggles
  document.querySelectorAll('.explain-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => toggleExplain(toggle));
  });

  // Hide all results panels initially
  document.querySelectorAll('[id$="-results"]').forEach(el => {
    el.style.display = 'none';
  });

  // Animate stat counters
  document.querySelectorAll('.stat-badge .num').forEach(el => {
    const target = parseInt(el.textContent);
    if (!isNaN(target)) {
      let current = 0;
      const step = target / 40;
      const interval = setInterval(() => {
        current = Math.min(current + step, target);
        el.textContent = Math.round(current) + (el.dataset.suffix || '');
        if (current >= target) clearInterval(interval);
      }, 25);
    }
  });

  console.log('🏥 Medical AI Platform — 13 tools loaded');
  console.log('📋 Based on medical-cv-builder skill');
  console.log('🔬 Stack: OpenCV → scikit-learn → Streamlit (simulated in JS)');
});

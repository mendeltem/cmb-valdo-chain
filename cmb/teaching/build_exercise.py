"""Build a self-contained classroom exercise: "segment the microbleeds yourself".

Students scroll through a short stack of T2*-weighted slices, paint what they believe are cerebral
microbleeds, press *Check*, and get their personal scores -- voxel Dice, lesion-wise F1,
sensitivity, precision and (deliberately) voxel specificity, which is ~1.0 for everybody and makes
the point that specificity is useless for tiny lesions. Each case allows three attempts. After every
attempt their own marks are coloured (green = on a microbleed, red = false alarm); the microbleeds
they missed are revealed after the third attempt or on request. The scores of all attempts stay
visible at the top, so they can watch themselves improve.

Everything -- images, reference masks, code -- is embedded in ONE html file: it can be handed out
on a USB stick or a learning platform and runs offline on laptops and tablets (pointer events).

Data: VALDO challenge 2021, Task 2 (public, CC BY-NC 4.0 -- non-commercial teaching is fine,
attribution is printed in the page). Private patient data must never be used here.
The reference masks sit in the file as plain PNGs; a curious student can extract them. For a
classroom exercise that is acceptable, for an exam it would not be.

    python -m cmb.teaching.build_exercise --out ergebnisse/lehre/cmb_exercise.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from typing import Dict, List

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = "/home/uchralt/data/work/valdo-t2s"

# Hand-picked from the 4 mm cohort: few slices, each microbleed visible in one or two of them.
# (case, first slice, number of slices, difficulty key)
CASES = [("sub-111", 9, 12, "easy"), ("sub-108", 14, 12, "medium"), ("sub-101", 6, 12, "hard")]


def png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_case(case_id: str, first: int, count: int, difficulty: str) -> Dict:
    folder = f"{ROOT}/dev/preproc_roh/{case_id}"          # bias-corrected, NOT vessel-inpainted
    nifti = nib.load(f"{folder}/{case_id}_image.nii.gz")
    inverted = np.asarray(nifti.dataobj, dtype=np.float32)   # stored as 1 - image / max, 0 outside the brain
    label = np.asarray(nib.load(f"{folder}/{case_id}_label.nii.gz").dataobj) > 0
    brain = inverted > 0
    shown = np.where(brain, 1.0 - inverted, 0.0)             # as acquired: microbleeds are dark

    slab = slice(first, first + count)
    xs, ys = np.nonzero(brain[:, :, slab].any(axis=2))
    x0, x1, y0, y1 = max(xs.min() - 2, 0), xs.max() + 3, max(ys.min() - 2, 0), ys.max() + 3   # tight: no wasted black border on a phone
    shown, label, brain = (v[x0:x1, y0:y1, slab] for v in (shown, label, brain))

    lo, hi = np.percentile(shown[brain], [1, 99.7])
    grey = np.clip((shown - lo) / (hi - lo), 0, 1)
    grey = np.where(brain, 1 + grey * 254, 0).astype(np.uint8)      # 0 is reserved for "outside the brain"

    # np.rot90 turns the (x, y) array so that anterior is up, as radiologists expect.
    images = [png_base64(np.rot90(grey[:, :, k])) for k in range(count)]
    masks = [png_base64(np.rot90(label[:, :, k]).astype(np.uint8) * 255) for k in range(count)]
    height, width = np.rot90(grey[:, :, 0]).shape
    zooms = [float(z) for z in nifti.header.get_zooms()[:3]]
    return dict(id=case_id, difficulty=difficulty, width=int(width), height=int(height), slices=count,
                voxel_mm=[round(zooms[0], 2), round(zooms[1], 2), round(zooms[2], 1)],
                images=images, masks=masks)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Microbleed exercise</title>
<style>
 :root{--bg:#000;--panel:#15181e;--line:#2b303a;--text:#eee;--dim:#9aa3b2;--accent:#4d8fe0}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent} html,body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}
 #app{max-width:900px;margin:0 auto;padding:0 6px}
 .bar{display:flex;gap:6px;align-items:center;padding:6px 0} .grow{flex:1}
 button,select{background:#262b35;color:var(--text);border:1px solid #454b58;border-radius:8px;padding:9px 11px;font-size:15px;min-height:40px}
 button.on{background:var(--accent);border-color:#79b0f0} button.go{background:#2e7d4f;border-color:#49a36f;font-weight:700;padding:9px 16px} button:disabled{opacity:.35}
 #stage{display:flex;justify-content:center;touch-action:none;position:relative} #zoomReset{position:absolute;right:6px;top:6px;display:none;min-height:0;padding:4px 9px;font-size:13px;opacity:.85} canvas{display:block;touch-action:none;cursor:crosshair}
 input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:34px;background:transparent;margin:0}
 input[type=range]::-webkit-slider-runnable-track{height:10px;border-radius:5px;background:#3a4150}
 input[type=range]::-moz-range-track{height:10px;border-radius:5px;background:#3a4150}
 input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:30px;height:30px;border-radius:50%;background:var(--accent);margin-top:-10px;border:2px solid #cfe2ff}
 input[type=range]::-moz-range-thumb{width:28px;height:28px;border-radius:50%;background:var(--accent);border:2px solid #cfe2ff}
 #sliceRow{display:flex;gap:8px;align-items:center} #sliceVal{min-width:54px;text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
 #result{font-size:14px;padding:2px 2px 4px;min-height:22px} #metrics{font-size:12.5px;line-height:1.45;padding:0 2px 4px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden} #metrics b{font-weight:600}
 .chip{display:inline-block;border:1px solid #555;border-radius:10px;padding:0 8px;margin:0 2px;cursor:pointer} .dim{color:var(--dim)} .tp{color:#6fdc8c}.fp{color:#ff7b72}.fn{color:#ffd54f}
 details{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 10px} summary{cursor:pointer;color:var(--dim);padding:4px 0}
 table{border-collapse:collapse;width:100%;font-size:14px} th,td{border-bottom:1px solid var(--line);padding:5px 4px;text-align:right} th:first-child,td:first-child{text-align:left} th{color:var(--dim);font-weight:500}
 ul{margin:6px 0 4px 18px;padding:0} li{margin:3px 0} .jump{text-decoration:underline;cursor:pointer;padding:0 4px} .opt{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;align-items:center}
 footer{color:var(--dim);font-size:12px;padding:8px 2px 16px}
</style></head><body><div id="app">
 <div class="bar" id="top">
  <select id="caseSel"></select>
  <span id="attemptInfo" class="dim grow" style="font-size:13px;text-align:right"></span>
 </div>
 <div id="metrics"></div>
 <div id="stage"><canvas id="view"></canvas><button id="zoomReset" title="reset zoom">1:1</button></div>
 <div id="sliceRow"><input id="slice" type="range" min="0" value="0"><span id="sliceVal"></span></div>
 <div class="bar" id="tools">
  <button id="mPaint" class="on" title="paint">&#9998;</button><button id="mErase" title="erase">&#9003;</button><button id="mMove" title="move the zoomed image">&#9995;</button><button id="undo" title="undo">&#8630;</button>
  <input id="brush" type="range" min="1" max="8" value="3" style="flex:1;min-width:60px" title="brush size">
  <button id="check" class="go"></button>
 </div>
 <div id="result" class="dim"></div>
 <details id="dScores"><summary data-t="sumScores"></summary><table id="scores"></table></details>
 <details><summary data-t="sumMore"></summary>
  <div class="opt"><span data-t="bright"></span><input id="bright" type="range" min="0.5" max="2" step="0.05" value="1">
   <span data-t="contrast"></span><input id="contr" type="range" min="0.5" max="3" step="0.05" value="1"></div>
  <div class="bar"><button id="clearSlice"></button><button id="reveal"></button><button id="restart"></button><span class="grow"></span><button id="lang">DE</button></div>
 </details>
 <details><summary data-t="howTitle"></summary><div data-t="howText"></div></details>
 <details><summary data-t="metTitle"></summary><div data-t="metText"></div></details>
 <footer data-t="credit"></footer>
</div>
<script>
"use strict";
const CASES = __CASES__;
const MAX_ATTEMPTS = 3;
const RENDER_SCALE = 2;      // internal canvas resolution relative to the image; the CSS size adapts to the screen

/* ---------------------------------------------------------------- texts (German / English) */
const TEXT = {
 de: { check:"Prüfen", reveal:"Lösung zeigen", restart:"Fall neu starten", paint:"Malen", erase:"Radieren", clearSlice:"Schicht leeren", brush:"Pinsel", bright:"Helligkeit", contrast:"Kontrast",
  easy:"leicht", medium:"mittel", hard:"schwer", caseWord:"Fall", attempt:"Versuch", of:"von", noMore:"3 Versuche verbraucht", solved:"Lösung sichtbar", slice:"Schicht",
  start:"Blutungen anmalen, dann unten rechts „Prüfen“. Schichten mit dem Balken wechseln – alle ansehen! Zwei Finger = zoomen und verschieben.", missedIn:"übersehen (gelber Ring) in Schicht", falseIn:"Fehlalarm (roter Ring) in Schicht", found:"gefunden", falseAl:"Fehlalarme", missed:"übersehen",
  fpSlices:"Fehlalarme in Schicht", fnSlices:"Übersehen in Schicht", fnHidden:"Wo die übersehenen liegen, zeigt der dritte Versuch (gelb).", perfect:"Alles gefunden, kein Fehlalarm!", nothing:"Noch nichts markiert.",
  legend:"grün = richtig · rot = Fehlalarm · gelb = Referenz, die du nicht getroffen hast", legendFn:"", sumScores:"Tabelle aller Versuche", sumMore:"Mehr: Helligkeit, Lösung, Neustart, Sprache",
  cols:["#","Dice","F1","Sens.","Prec.","Spez.","✓ / ✗ / ?"],
  howTitle:"Woran erkennt man eine Mikroblutung?",
  howText:"<ul><li><b>dunkel, rund oder oval, 2–10 mm</b>, scharf begrenzt</li><li>auf T2* größer als in Wirklichkeit (Blooming)</li><li>endet nach ein bis zwei Schichten – ein <b>Gefäß</b> läuft weiter oder ist länglich</li><li><b>Verwechslungen:</b> Gefäße in den Furchen, Verkalkungen und Eisen in den Basalganglien (oft symmetrisch), Artefakte an Knochen und Luft, Partialvolumen am Hirnrand</li></ul>",
  metTitle:"Was bedeuten die Zahlen?",
  metText:"<ul><li><b>Dice</b>: Überlappung deiner Fläche mit der Referenz, Voxel für Voxel. 1 = identisch. Bei winzigen Objekten kosten schon ein, zwei Voxel daneben viel.</li><li><b>Sensitivität</b>: Anteil der echten Blutungen, die du getroffen hast (berühren genügt).</li><li><b>Precision</b>: Anteil deiner Markierungen, die wirklich eine Blutung treffen.</li><li><b>F1</b>: Mittel aus beiden – das übliche Maß für Detektion.</li><li><b>Spezifität</b>: Anteil der gesunden Voxel, die du in Ruhe gelassen hast. Sie ist bei <i>jedem</i> fast 1,0, weil 99,99 % des Hirns gesund sind – deshalb sagt sie hier nichts aus.</li></ul>",
  credit:"Bilddaten: VALDO Challenge 2021, Task 2 (Sudre et al., Medical Image Analysis 2024), CC BY-NC 4.0 – nur nicht-kommerzielle Lehre. Referenz = Annotation der Challenge; auch Experten sind sich nicht immer einig." },
 en: { check:"Check", reveal:"Show solution", restart:"Restart case", paint:"Paint", erase:"Erase", clearSlice:"Clear slice", brush:"Brush", bright:"Brightness", contrast:"Contrast",
  easy:"easy", medium:"medium", hard:"hard", caseWord:"Case", attempt:"Attempt", of:"of", noMore:"3 attempts used", solved:"solution shown", slice:"Slice",
  start:"Paint the microbleeds, then press “Check” (bottom right). Change slices with the bar – look at all of them! Two fingers = zoom and move.", missedIn:"missed (yellow ring) in slice", falseIn:"false alarm (red ring) in slice", found:"found", falseAl:"false alarms", missed:"missed",
  fpSlices:"False alarms in slice", fnSlices:"Missed in slice", fnHidden:"Where the missed ones are is shown after the third attempt (yellow).", perfect:"All found, no false alarm!", nothing:"Nothing marked yet.",
  legend:"green = correct · red = false alarm · yellow = reference you did not cover", legendFn:"", sumScores:"Table of all attempts", sumMore:"More: brightness, solution, restart, language",
  cols:["#","Dice","F1","Sens.","Prec.","Spec.","✓ / ✗ / ?"],
  howTitle:"What does a microbleed look like?",
  howText:"<ul><li><b>dark, round or oval, 2–10 mm</b>, sharply delineated</li><li>looks larger on T2* than it is (blooming)</li><li>ends after one or two slices – a <b>vessel</b> continues or is elongated</li><li><b>Mimics:</b> vessels in the sulci, calcification and iron in the basal ganglia (often symmetric), artefacts near bone and air, partial volume at the brain surface</li></ul>",
  metTitle:"What do the numbers mean?",
  metText:"<ul><li><b>Dice</b>: overlap of your area with the reference, voxel by voxel. 1 = identical. For tiny objects one or two voxels off already cost a lot.</li><li><b>Sensitivity</b>: share of true microbleeds you hit (touching is enough).</li><li><b>Precision</b>: share of your marks that really hit a microbleed.</li><li><b>F1</b>: mean of the two – the usual score for detection.</li><li><b>Specificity</b>: share of healthy voxels you left alone. It is ~1.0 for <i>everybody</i>, because 99.99 % of the brain is healthy – which is why it tells you nothing here.</li></ul>",
  credit:"Images: VALDO challenge 2021, Task 2 (Sudre et al., Medical Image Analysis 2024), CC BY-NC 4.0 – non-commercial teaching only. Reference = challenge annotation; even experts do not always agree." }
};
let lang = 'en';
const t = key => TEXT[lang][key];
const $ = id => document.getElementById(id);

/* ---------------------------------------------------------------- state */
const view = $('view'), ctx = view.getContext('2d');
const work = document.createElement('canvas'), wctx = work.getContext('2d', {willReadFrequently:true});
let C = null;          // current case: decoded images, reference, brain mask, the student's paint, attempts
const store = {};      // one entry per case id, kept while the page is open
let mode = 'paint', painting = false, stroke = null;
// View transform in canvas pixels (pinch zoom / two-finger pan on touch screens, Ctrl + wheel on a desktop).
const V = {s:1, x:0, y:0}, touches = new Map(); let pinch = null, pan = null;

function decode(b64){ return new Promise(resolve => { const im = new Image(); im.onload = () => resolve(im); im.src = 'data:image/png;base64,' + b64; }); }
function pixels(im){ work.width = im.naturalWidth; work.height = im.naturalHeight; wctx.drawImage(im, 0, 0); return wctx.getImageData(0, 0, work.width, work.height).data; }

async function openCase(index){
  const meta = CASES[index];
  if (!store[meta.id]){
    const n = meta.width * meta.height, entry = { meta, images:[], reference:new Uint8Array(n * meta.slices), brain:new Uint8Array(n * meta.slices),
      paint:new Uint8Array(n * meta.slices), undo:[], attempts:[], revealed:false, feedback:null, z:Math.floor(meta.slices / 2) };
    for (let k = 0; k < meta.slices; k++){
      const im = await decode(meta.images[k]); entry.images.push(im);
      const grey = pixels(im); for (let i = 0; i < n; i++) entry.brain[k * n + i] = grey[4 * i] > 0 ? 1 : 0;
      const ref = pixels(await decode(meta.masks[k])); for (let i = 0; i < n; i++) entry.reference[k * n + i] = ref[4 * i] > 127 ? 1 : 0;
    }
    store[meta.id] = entry;
  }
  C = store[meta.id]; V.s = 1; V.x = 0; V.y = 0;
  $('slice').max = C.meta.slices - 1; $('slice').value = C.z;
  fit(); render(); renderScores(); renderResult(); renderButtons();
}

/* ---------------------------------------------------------------- layout: the whole brain and all controls fit the screen */
function fit(){
  if (!C) return;
  const m = C.meta, fixed = $('top').offsetHeight + $('metrics').offsetHeight + $('sliceRow').offsetHeight + $('tools').offsetHeight + $('result').offsetHeight + 14;
  const maxW = $('app').clientWidth, maxH = Math.max(window.innerHeight - fixed, 220);
  const w = Math.min(maxW, maxH * m.width / m.height);
  view.style.width = Math.floor(w) + 'px'; view.style.height = Math.floor(w * m.height / m.width) + 'px';
}
window.addEventListener('resize', fit); window.addEventListener('orientationchange', () => setTimeout(fit, 200));

/* ---------------------------------------------------------------- drawing */
function render(){
  if (!C) return;
  const m = C.meta, n = m.width * m.height, off = C.z * n;
  view.width = m.width * RENDER_SCALE; view.height = m.height * RENDER_SCALE;      // also clears the canvas
  ctx.setTransform(V.s, 0, 0, V.s, V.x, V.y);
  ctx.imageSmoothingEnabled = true;
  ctx.filter = 'brightness(' + $('bright').value + ') contrast(' + $('contr').value + ')';
  ctx.drawImage(C.images[C.z], 0, 0, view.width, view.height); ctx.filter = 'none';
  work.width = m.width; work.height = m.height;      // overlay at native resolution, scaled without smoothing
  const over = wctx.createImageData(m.width, m.height), d = over.data, fb = C.feedback;
  for (let i = 0; i < n; i++){
    let r = 0, g = 0, b = 0, a = 0;
    if (C.paint[off + i]){
      if (fb){ if (fb.voxelClass[off + i] === 1){ r = 60; g = 220; b = 110; } else { r = 255; g = 70; b = 70; } a = 170; }
      else { r = 80; g = 170; b = 255; a = 150; }
    } else if ((C.revealed || fb) && C.reference[off + i]){ r = 255; g = 213; b = 79; a = 210; }   // reference the student did not cover
    const p = 4 * i; d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = a;
  }
  wctx.putImageData(over, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(work, 0, 0, view.width, view.height);
  if (fb){ ctx.lineWidth = 2.5 / Math.sqrt(V.s); for (const mark of fb.marks){ if (mark.z !== C.z) continue;
      ctx.strokeStyle = mark.kind === 'missed' ? '#ffd54f' : '#ff5a5a'; ctx.beginPath();
      ctx.arc((mark.x + 0.5) * RENDER_SCALE, (mark.y + 0.5) * RENDER_SCALE, 11 * RENDER_SCALE, 0, 2 * Math.PI); ctx.stroke(); } }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  $('zoomReset').style.display = V.s > 1.01 ? 'block' : 'none';
  $('sliceVal').textContent = (C.z + 1) + ' / ' + m.slices;
}

function paintAt(event){
  const m = C.meta, box = view.getBoundingClientRect();
  const px = (event.clientX - box.left) * view.width / box.width, py = (event.clientY - box.top) * view.height / box.height;   // canvas pixels
  const cx = Math.floor((px - V.x) / V.s / RENDER_SCALE), cy = Math.floor((py - V.y) / V.s / RENDER_SCALE);
  const radius = +$('brush').value - 1, n = m.width * m.height, value = mode === 'paint' ? 1 : 0;
  for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++){
    if (dx * dx + dy * dy > radius * radius + 0.5) continue;
    const x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= m.width || y >= m.height) continue;
    const i = C.z * n + y * m.width + x; if (!C.brain[i] || C.paint[i] === value) continue;
    stroke.push(i, C.paint[i]); C.paint[i] = value;
  }
  C.feedback = null;               // the colours of the last check no longer match the painting
  render();
}
function canvasPoint(e){ const box = view.getBoundingClientRect(); return {x:(e.clientX - box.left) * view.width / box.width, y:(e.clientY - box.top) * view.height / box.height}; }
function clampView(){ V.s = Math.max(1, Math.min(8, V.s)); V.x = Math.min(0, Math.max(view.width * (1 - V.s), V.x)); V.y = Math.min(0, Math.max(view.height * (1 - V.s), V.y)); }
function zoomAbout(point, factor){ const s0 = V.s; V.s = Math.max(1, Math.min(8, V.s * factor)); V.x = point.x - (point.x - V.x) * V.s / s0; V.y = point.y - (point.y - V.y) * V.s / s0; clampView(); render(); }
function abortStroke(){ if (stroke) for (let k = stroke.length - 2; k >= 0; k -= 2) C.paint[stroke[k]] = stroke[k + 1]; painting = false; stroke = null; }
view.addEventListener('pointerdown', e => { if (!C) return; e.preventDefault(); view.setPointerCapture(e.pointerId); touches.set(e.pointerId, canvasPoint(e));
  if (touches.size === 2){            // second finger: this is a zoom gesture, not a brush stroke -- take back what the first finger painted
    abortStroke(); const [a, b] = [...touches.values()];
    pinch = {d:Math.hypot(a.x - b.x, a.y - b.y) || 1, mx:(a.x + b.x) / 2, my:(a.y + b.y) / 2, s:V.s, x:V.x, y:V.y}; render(); return; }
  if (touches.size === 1 && !pinch){ if (mode === 'move'){ const c = canvasPoint(e); pan = {px:c.x, py:c.y, x:V.x, y:V.y}; } else { painting = true; stroke = []; paintAt(e); } } });
view.addEventListener('pointermove', e => { if (!touches.has(e.pointerId)) return; e.preventDefault(); touches.set(e.pointerId, canvasPoint(e));
  if (pinch && touches.size >= 2){ const [a, b] = [...touches.values()], d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    V.s = Math.max(1, Math.min(8, pinch.s * d / pinch.d));
    V.x = mx - (pinch.mx - pinch.x) * V.s / pinch.s; V.y = my - (pinch.my - pinch.y) * V.s / pinch.s; clampView(); render(); return; }
  if (pan && touches.size === 1){ const c = canvasPoint(e); V.x = pan.x + c.x - pan.px; V.y = pan.y + c.y - pan.py; clampView(); render(); return; }
  if (painting) paintAt(e); });
function endPointer(e){ touches.delete(e.pointerId); pan = null; if (painting && stroke && stroke.length) C.undo.push(stroke); painting = false; stroke = null;
  if (touches.size === 0) pinch = null; }      // the gesture ends only when ALL fingers are up, so the remaining finger does not paint
view.addEventListener('pointerup', endPointer); view.addEventListener('pointercancel', endPointer);
$('zoomReset').onclick = () => { V.s = 1; V.x = 0; V.y = 0; render(); };
$('stage').addEventListener('wheel', e => { if (!C) return; e.preventDefault(); if (e.ctrlKey) zoomAbout(canvasPoint(e), e.deltaY > 0 ? 1 / 1.2 : 1.2); else setSlice(C.z + (e.deltaY > 0 ? 1 : -1)); }, {passive:false});
function setSlice(z){ C.z = Math.max(0, Math.min(C.meta.slices - 1, z)); $('slice').value = C.z; render(); }

/* ---------------------------------------------------------------- scoring */
// 26-connected components of a binary volume; returns the label volume and the number of components.
function components(mask, m){
  const n = m.width * m.height, labels = new Int32Array(mask.length); let count = 0; const stack = [];
  for (let start = 0; start < mask.length; start++){
    if (!mask[start] || labels[start]) continue;
    labels[start] = ++count; stack.push(start);
    while (stack.length){
      const i = stack.pop(), z = Math.floor(i / n), y = Math.floor((i - z * n) / m.width), x = i - z * n - y * m.width;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++){
        const X = x + dx, Y = y + dy, Z = z + dz; if (X < 0 || Y < 0 || Z < 0 || X >= m.width || Y >= m.height || Z >= m.slices) continue;
        const j = Z * n + Y * m.width + X; if (mask[j] && !labels[j]){ labels[j] = count; stack.push(j); }
      }
    }
  }
  return {labels, count};
}
function score(){
  const m = C.meta, n = m.width * m.height, P = C.paint, R = C.reference;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < P.length; i++){ if (!C.brain[i]) continue; if (P[i]) { if (R[i]) tp++; else fp++; } else { if (R[i]) fn++; else tn++; } }
  const mine = components(P, m), truth = components(R, m);
  // Lesion matching is tolerant: a mark that comes within TOLERANCE voxels (in-plane, ~1 mm) of a microbleed counts
  // as having found it -- a finger on a phone is not a scalpel. The voxel-wise Dice above stays strict.
  const TOLERANCE = 2, near = new Int32Array(P.length);
  for (let i = 0; i < R.length; i++){ if (!R[i]) continue; const z = Math.floor(i / n), y = Math.floor((i - z * n) / m.width), x = i - z * n - y * m.width;
    for (let dy = -TOLERANCE; dy <= TOLERANCE; dy++) for (let dx = -TOLERANCE; dx <= TOLERANCE; dx++){ const X = x + dx, Y = y + dy;
      if (X < 0 || Y < 0 || X >= m.width || Y >= m.height) continue; const j = z * n + Y * m.width + X; if (!near[j] || R[j]) near[j] = truth.labels[i]; } }
  const mineHits = new Uint8Array(mine.count + 1), truthHit = new Uint8Array(truth.count + 1);
  for (let i = 0; i < P.length; i++) if (P[i] && near[i]){ mineHits[mine.labels[i]] = 1; truthHit[near[i]] = 1; }
  let found = 0; for (let k = 1; k <= truth.count; k++) found += truthHit[k];
  let falseAlarms = 0; for (let k = 1; k <= mine.count; k++) falseAlarms += mineHits[k] ? 0 : 1;
  const missed = truth.count - found, div = (a, b) => b ? a / b : 0;
  // per voxel: 1 = belongs to a mark that touches a microbleed, 2 = belongs to a false alarm
  const voxelClass = new Uint8Array(P.length); for (let i = 0; i < P.length; i++) if (P[i]) voxelClass[i] = mineHits[mine.labels[i]] ? 1 : 2;
  // one ring per (component, slice): centre of the component's voxels in that slice
  const marks = [], ringsOf = (labels, flags, kind) => { const acc = new Map();
    for (let i = 0; i < labels.length; i++){ const l = labels[i]; if (!l || flags[l]) continue; const z = Math.floor(i / n), key = l * 1000 + z, y = Math.floor((i - z * n) / m.width), x = i - z * n - y * m.width;
      const a = acc.get(key) || {z, sx:0, sy:0, c:0}; a.sx += x; a.sy += y; a.c++; acc.set(key, a); }
    acc.forEach(a => marks.push({kind, z:a.z, x:a.sx / a.c, y:a.sy / a.c})); };
  ringsOf(truth.labels, truthHit, 'missed'); ringsOf(mine.labels, mineHits, 'false');
  const slicesOf = (labels, flags, want) => { const s = new Set(); for (let i = 0; i < labels.length; i++) if (labels[i] && flags[labels[i]] === want) s.add(Math.floor(i / n)); return [...s].sort((a, b) => a - b); };
  return { dice: div(2 * tp, 2 * tp + fp + fn), f1: div(2 * found, 2 * found + falseAlarms + missed), sensitivity: div(found, truth.count),
           precision: div(found, found + falseAlarms), specificity: div(tn, tn + fp), found, falseAlarms, missed, voxelClass, marks,
           fpSlices: slicesOf(mine.labels, mineHits, 0), fnSlices: slicesOf(truth.labels, truthHit, 0), empty: mine.count === 0 };
}

/* ---------------------------------------------------------------- panels */
const dec = (v, k) => v.toFixed(k).replace('.', lang === 'de' ? ',' : '.');
function renderScores(){
  const rows = C ? C.attempts : [];
  let html = '<tr>' + t('cols').map(c => '<th>' + c + '</th>').join('') + '</tr>';
  for (let k = 0; k < MAX_ATTEMPTS; k++){
    const a = rows[k];
    html += '<tr><td>' + (k + 1) + '</td>' + (a ? [a.dice, a.f1, a.sensitivity, a.precision].map(v => '<td>' + dec(v, 2) + '</td>').join('') +
      '<td>' + dec(a.specificity, 4) + '</td><td><span class="tp">' + a.found + '</span> / <span class="fp">' + a.falseAlarms + '</span> / <span class="fn">' + a.missed + '</span></td>'
      : '<td class="dim">–</td>'.repeat(6)) + '</tr>';
  }
  $('scores').innerHTML = html;
}
// Metrics of every attempt stay on top of the image; hints about WHERE things went wrong go below the tools.
function renderResult(){
  if (!C) return; const rows = C.attempts, top = $('metrics'), el = $('result');
  if (!rows.length){ top.innerHTML = ''; el.className = 'dim'; el.textContent = t('start'); fit(); return; }
  top.innerHTML = rows.map((a, k) => '<div><span class="dim">' + (k + 1) + '</span> &nbsp;<b>Dice ' + dec(a.dice, 2) + '</b> · <b>F1 ' + dec(a.f1, 2) + '</b> · Sens ' + dec(a.sensitivity, 2) +
    ' · Prec ' + dec(a.precision, 2) + ' · Spec ' + dec(a.specificity, 3) + ' &nbsp;<span class="tp">✓' + a.found + '</span> <span class="fp">✗' + a.falseAlarms + '</span> <span class="fn">?' + a.missed + '</span></div>').join('');
  const a = rows[rows.length - 1], chips = (list, cls) => list.map(z => '<span class="chip ' + cls + '" data-z="' + z + '">' + (z + 1) + '</span>').join('');
  let h = '';
  if (a.empty) h = t('nothing');
  else if (!a.falseAlarms && !a.missed) h = '<b class="tp">' + t('perfect') + '</b>';
  else {
    if (a.falseAlarms) h += '<span class="fp">' + t('falseIn') + '</span> ' + chips(a.fpSlices, 'fp') + ' ';
    if (a.missed) h += '<span class="fn">' + t('missedIn') + '</span> ' + chips(a.fnSlices, 'fn');
  }
  h += '<div class="dim" style="font-size:12px">' + t('legend') + '</div>';
  el.className = ''; el.innerHTML = h;
  el.querySelectorAll('.chip').forEach(c => c.onclick = () => setSlice(+c.dataset.z));
  fit();
}
function renderButtons(){
  const used = C ? C.attempts.length : 0, done = C && (used >= MAX_ATTEMPTS || C.revealed);
  $('check').disabled = !!done; $('reveal').disabled = !C || C.revealed;
  $('attemptInfo').textContent = !C ? '' : C.revealed ? t('solved') : used >= MAX_ATTEMPTS ? t('noMore') : t('attempt') + ' ' + (used + 1) + ' ' + t('of') + ' ' + MAX_ATTEMPTS;
  $('mPaint').classList.toggle('on', mode === 'paint'); $('mErase').classList.toggle('on', mode === 'erase'); $('mMove').classList.toggle('on', mode === 'move');
  view.style.cursor = mode === 'move' ? 'grab' : 'crosshair';
}
function applyLanguage(){
  document.documentElement.lang = lang; $('lang').textContent = lang === 'de' ? 'EN' : 'DE';
  document.querySelectorAll('[data-t]').forEach(el => el.innerHTML = t(el.dataset.t));
  [['check','check'],['reveal','reveal'],['restart','restart'],['clearSlice','clearSlice']].forEach(([id, key]) => $(id).textContent = t(key));
  $('mPaint').title = t('paint'); $('mErase').title = t('erase');      // icons only: the tool row must fit a phone
  const sel = $('caseSel'), keep = sel.selectedIndex; sel.innerHTML = '';
  CASES.forEach((c, k) => { const o = document.createElement('option'); o.textContent = t('caseWord') + ' ' + String.fromCharCode(65 + k) + ' · ' + t(c.difficulty); sel.appendChild(o); });
  sel.selectedIndex = Math.max(keep, 0); renderScores(); renderResult(); renderButtons();
}
function afterCheck(){ render(); renderScores(); renderResult(); renderButtons(); }

/* ---------------------------------------------------------------- controls */
$('check').onclick = () => { if (!C || C.attempts.length >= MAX_ATTEMPTS) return; const s = score(); C.attempts.push(s); C.feedback = s; if (C.attempts.length >= MAX_ATTEMPTS) C.revealed = true; afterCheck(); };
$('reveal').onclick = () => { if (!C) return; if (!C.attempts.length){ const s = score(); C.attempts.push(s); C.feedback = s; } C.revealed = true; afterCheck(); };
$('restart').onclick = () => { if (!C) return; C.paint.fill(0); C.undo = []; C.attempts = []; C.revealed = false; C.feedback = null; afterCheck(); };
$('undo').onclick = () => { const s = C && C.undo.pop(); if (!s) return; for (let k = s.length - 2; k >= 0; k -= 2) C.paint[s[k]] = s[k + 1]; C.feedback = null; render(); };
$('clearSlice').onclick = () => { if (!C) return; const n = C.meta.width * C.meta.height, s = []; for (let i = C.z * n; i < (C.z + 1) * n; i++) if (C.paint[i]){ s.push(i, 1); C.paint[i] = 0; } if (s.length) C.undo.push(s); C.feedback = null; render(); };
$('mPaint').onclick = () => { mode = 'paint'; renderButtons(); }; $('mErase').onclick = () => { mode = 'erase'; renderButtons(); }; $('mMove').onclick = () => { mode = 'move'; renderButtons(); };
$('slice').oninput = e => setSlice(+e.target.value);
['bright','contr'].forEach(id => $(id).oninput = render);
$('caseSel').onchange = e => openCase(e.target.selectedIndex);
$('lang').onclick = () => { lang = lang === 'de' ? 'en' : 'de'; applyLanguage(); };
window.addEventListener('keydown', e => { if (!C) return; if (e.key === 'ArrowUp' || e.key === 'ArrowRight') setSlice(C.z + 1); else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') setSlice(C.z - 1);
  else if (e.key === 'e') { mode = 'erase'; renderButtons(); } else if (e.key === 'p') { mode = 'paint'; renderButtons(); } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) $('undo').click(); else return; e.preventDefault(); });

applyLanguage();
openCase(0).then(() => {
  // Self-test for screenshots: "page.html#demo" marks the first microbleed slightly off-centre, adds one false
  // alarm and presses Check, so the feedback layout can be inspected without a finger.
  if (location.hash === '#demozoom'){
    const m = C.meta, n = m.width * m.height, first = C.reference.indexOf(1), z = Math.floor(first / n), y = Math.floor((first - z * n) / m.width), x = first - z * n - y * m.width;
    setSlice(z); zoomAbout({x:(x + 0.5) * RENDER_SCALE, y:(y + 0.5) * RENDER_SCALE}, 3);
    const box = view.getBoundingClientRect(), toClient = (vx, vy) => ({clientX: box.left + ((vx + 0.5) * RENDER_SCALE * V.s + V.x) * box.width / view.width, clientY: box.top + ((vy + 0.5) * RENDER_SCALE * V.s + V.y) * box.height / view.height});
    stroke = []; paintAt(toClient(x, y)); stroke = null;
    document.title = C.paint[first] === 1 ? 'ZOOM-MAPPING OK' : 'ZOOM-MAPPING WRONG'; $('check').click(); return; }
  if (location.hash !== '#demo') return;
  const m = C.meta, n = m.width * m.height, first = C.reference.indexOf(1), z = Math.floor(first / n);
  const y = Math.floor((first - z * n) / m.width), x = first - z * n - y * m.width;
  for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++){ C.paint[z * n + (y + dy) * m.width + x + dx + 1] = 1; C.paint[z * n + (y + 60 + dy) * m.width + x + dx] = 1; }
  setSlice(z); $('check').click();
});
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="ergebnisse/lehre/cmb_exercise.html")
    args = parser.parse_args()
    out = args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cases: List[Dict] = [load_case(*c) for c in CASES]
    open(out, "w", encoding="utf-8").write(PAGE.replace("__CASES__", json.dumps(cases)))
    for c in cases:
        print(f"{c['id']} ({c['difficulty']}): {c['slices']} slices of {c['width']} x {c['height']} px, voxel {c['voxel_mm']} mm")
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

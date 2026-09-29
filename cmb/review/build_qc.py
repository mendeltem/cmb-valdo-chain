"""Quality-control page: every patient as a scrollable sheet of axial slices, reference and prediction side by side.

Modelled on the WMH editor ``7_qc_edit.py`` (same look, same conventions): one HTML file without a
server, a searchable / sortable patient list on the left, layer chips, keyboard navigation, and a
verdict per patient that is written to one JSON file (chosen folder or download).

What one sees per patient
-------------------------
* axial slices only, scrolling down from the lowest to the highest slice -- scrolling IS the z axis, and all columns
  scroll together so the same slice is always side by side. With ``--columns 3`` a first column shows the plain image
  without any outline (user request 2026-09-22: judging a slice is easier when one version carries no drawing at all).
  TWO columns per slice by default:
  left the image with the reference annotation (yellow), right the SAME image with the network
  prediction (red; fixed post-processing: threshold 0.3, at least 2 mm3). Drawing both outlines
  into one image proved confusing (user feedback 2026-09-20): a detected lesion shows two nearly
  identical outlines on top of each other. Side by side, agreement and disagreement are obvious;
* the image as acquired (microbleeds dark);
* the training grid has 1 mm slices although the scans were acquired with 0.8-5 mm. Showing every
  interpolated slice would be tedious, so slices are shown at roughly the ACQUIRED spacing, and
  each tile carries the outlines of all grid slices it stands for -- no finding can hide between
  two displayed slices;
* tiles with a false alarm, a missed or a detected microbleed get a coloured frame, and the list
  shows the counts, so one can jump straight to the problem cases.

Per patient the page needs three files next to it: ``<id>.jpg`` (slices), ``<id>_ref.png`` and
``<id>_pred.png`` (transparent outlines). Private cohorts: build into a private folder, never send.

    python -m cmb.review.build_qc --dataset VALDO --manifest dev/manifest_valdo_gitter_d.json \
        --predictions ergebnisse/turnier/a03-aniso-e60-gd --out ergebnisse/qc/valdo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/home/uchralt/data/work/valdo-t2s"
sys.path.insert(0, f"{ROOT}/code")
CONNECTIVITY_26 = np.ones((3, 3, 3), bool)
THRESHOLD, MIN_MM3, MAX_VOXELS, VOXEL_MM3 = 0.3, 2.0, 2100, 0.25
COLUMNS, TILE_WIDTH, MAX_TILES = 1, 520, 66      # ONE slice per row in the image file; the page shows it twice, side by side


def post_process(probability: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(probability > THRESHOLD, structure=CONNECTIVITY_26)
    size = np.bincount(labels.ravel(), minlength=n + 1)
    keep = (size * VOXEL_MM3 >= MIN_MM3) & (size <= MAX_VOXELS); keep[0] = False
    return keep[labels]


def outline(mask: np.ndarray, size) -> np.ndarray:
    big = np.asarray(Image.fromarray(np.rot90(mask).astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return big & ~ndimage.binary_erosion(big, iterations=1)


def acquired_slice_mm(case: Dict) -> float:
    """Slice spacing of the ORIGINAL scan (the grid itself is always 1 mm)."""
    info = f"{case['dir']}/info.json"
    if os.path.exists(info):
        spacing = json.load(open(info)).get("quelle_spacing")
        if spacing:
            return float(spacing[2])
    return 1.0


def build_patient(case: Dict, prediction_dir: Optional[str], out_dir: str) -> Dict:
    cid = case["id"]
    load = lambda p: np.asarray(nib.load(p).dataobj)
    inverted = load(f"{case['dir']}/{cid}_image.nii.gz").astype(np.float32)
    reference = load(f"{case['dir']}/{cid}_label.nii.gz") > 0
    prediction = None
    if prediction_dir:
        found = glob.glob(f"{prediction_dir}/fold*/{cid}_pred_proba.nii.gz")
        if found:
            prediction = post_process(load(found[0]).astype(np.float32) / 255.0)

    brain = inverted > 0
    shown = np.where(brain, 1.0 - inverted, 0.0)                 # as acquired: microbleeds dark
    lo, hi = np.percentile(shown[brain], [1, 98])      # 99.5 lets the bright CSF rim darken the tissue
    grey = (np.clip((shown - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    grey[~brain] = 0

    xs, ys = np.nonzero(brain.any(axis=2)); zs = np.nonzero(brain.any(axis=(0, 1)))[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    step = max(1, int(round(acquired_slice_mm(case))))
    while (zs.max() - zs.min() + 1) / step > MAX_TILES:
        step += 1
    centres = list(range(int(zs.min()) + step // 2, int(zs.max()) + 1, step))

    tile_w = TILE_WIDTH; tile_h = int(round(tile_w * (x1 - x0) / (y1 - y0)))      # rot90: rows come from x
    rows = -(-len(centres) // COLUMNS)
    sheet = Image.new("L", (COLUMNS * tile_w, rows * tile_h))
    layers = {name: np.zeros((rows * tile_h, COLUMNS * tile_w, 4), np.uint8) for name in ("ref", "pred")}
    colours = {"ref": (255, 214, 10, 255), "pred": (255, 69, 58, 255)}

    # lesion-level bookkeeping (26-connectivity, touch rule -- identical to metric.hits)
    ref_labels, n_ref = ndimage.label(reference, structure=CONNECTIVITY_26)
    tiles = []
    if prediction is not None:
        pred_labels, n_pred = ndimage.label(prediction, structure=CONNECTIVITY_26)
        hit_ref = np.zeros(n_ref + 1, bool); hit_pred = np.zeros(n_pred + 1, bool)
        both = reference & prediction
        hit_ref[np.unique(ref_labels[both])] = True; hit_pred[np.unique(pred_labels[both])] = True
        hit_ref[0] = hit_pred[0] = False
    for k, z in enumerate(centres):
        lo_z, hi_z = z - step // 2, z - step // 2 + step                          # grid slices this tile stands for
        r, c = divmod(k, COLUMNS)
        sheet.paste(Image.fromarray(np.rot90(grey[x0:x1, y0:y1, z])).resize((tile_w, tile_h), Image.BILINEAR), (c * tile_w, r * tile_h))
        box = (slice(r * tile_h, (r + 1) * tile_h), slice(c * tile_w, (c + 1) * tile_w))
        flags = dict(z=int(z), detected=0, missed=0, false=0)
        ref_here = reference[x0:x1, y0:y1, lo_z:hi_z].any(axis=2)
        if ref_here.any():
            layers["ref"][box][outline(ref_here, (tile_w, tile_h))] = colours["ref"]
        if prediction is not None:
            pred_here = prediction[x0:x1, y0:y1, lo_z:hi_z].any(axis=2)
            if pred_here.any():
                layers["pred"][box][outline(pred_here, (tile_w, tile_h))] = colours["pred"]
            ids = np.unique(ref_labels[x0:x1, y0:y1, lo_z:hi_z]); ids = ids[ids > 0]
            flags["detected"] = int(hit_ref[ids].sum()); flags["missed"] = int((~hit_ref[ids]).sum())
            pids = np.unique(pred_labels[x0:x1, y0:y1, lo_z:hi_z]); pids = pids[pids > 0]
            flags["false"] = int((~hit_pred[pids]).sum())
        else:
            flags["missed"] = int(len(np.unique(ref_labels[x0:x1, y0:y1, lo_z:hi_z])) - 1) if ref_here.any() else 0
        tiles.append(flags)

    sheet.save(f"{out_dir}/img/{cid}.jpg", quality=88)
    for name, layer in layers.items():
        Image.fromarray(layer, "RGBA").save(f"{out_dir}/img/{cid}_{name}.png", optimize=True)
    summary = dict(id=cid, cohort=case.get("cohort", ""), fold=case.get("fold"), n_ref=int(n_ref), tiles=tiles,
                   tile=[tile_w, tile_h], step_mm=step, has_prediction=prediction is not None)
    if prediction is not None:
        summary.update(tp=int(hit_ref.sum()), fn=int(n_ref - hit_ref.sum()), fp=int(n_pred - hit_pred.sum()))
    return summary


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CMB QC __DATASET__</title>
<style>
  :root{--bg:#080c0f;--panel:#0f161b;--line:#22363f;--ink:#e6eef2;--muted:#8aa0ab;--accent:#f0a830;--ref:#ffd60a;--pred:#ff453a;--ok:#42e07a}
  *{box-sizing:border-box} html,body{height:100%;margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 ui-sans-serif,system-ui,sans-serif}
  body{display:flex;flex-direction:column}
  header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
  h1{font-size:15px;margin:0} h1 .a{color:var(--accent)} .sub{color:var(--muted);font-size:11.5px} .grp{display:flex;gap:5px;align-items:center}
  .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  button{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:5px 9px;font:inherit;cursor:pointer;white-space:nowrap}
  button.on{border-color:var(--accent);color:var(--accent)} button.save{border-color:var(--ok);color:#dfffe9}
  input[type=range]{accent-color:var(--accent);width:96px} input[type=search],input[type=text]{width:100%;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font:inherit}
  main{flex:1;display:grid;grid-template-columns:270px 1fr;min-height:0}
  aside{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
  .tools{padding:8px 10px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px}
  .list{flex:1;overflow:auto;min-height:0}
  .row{padding:7px 10px;border-bottom:1px solid rgba(34,54,63,.55);cursor:pointer}
  .row:hover{background:rgba(240,168,48,.07)} .row.on{background:rgba(240,168,48,.15);border-left:3px solid var(--accent);padding-left:7px}
  .row .sid{font-weight:700;display:flex;justify-content:space-between;gap:8px} .row .sid em{font-style:normal;color:var(--muted);font-weight:400}
  .row .m{font-size:11px;color:var(--muted);margin-top:2px} .tp{color:var(--ok)}.fp{color:var(--pred)}.fn{color:var(--ref)}
  .pane{display:flex;flex-direction:column;min-height:0;background:#000}
  .phead{display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
  .chip{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border:1px solid var(--line);border-radius:7px;cursor:pointer;user-select:none;font-size:11.5px;color:var(--muted)}
  .chip.on{color:var(--ink);border-color:currentColor} .chip i{width:9px;height:9px;border-radius:2px;display:inline-block}
  #scroll{flex:1;overflow-y:auto;min-height:0} #sheet{display:grid;grid-template-columns:__GRID__;gap:4px;margin:0 auto;max-width:__MAXW__px}
  .col{position:relative} .col img{display:block;width:100%} .col img.layer{position:absolute;left:0;top:0}
  .colhead{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:__GRID__;gap:4px;max-width:__MAXW__px;margin:0 auto;background:var(--panel);border-bottom:1px solid var(--line)}
  .colhead div{padding:4px 10px;font-size:12px;font-weight:700} .colhead .r{color:var(--ref)} .colhead .p{color:var(--pred)}
  .tile{position:absolute;border:2px solid transparent;pointer-events:none} .tile > span{position:absolute;left:4px;top:3px;font-size:11.5px;color:#cfd8dc;background:rgba(8,12,15,.7);padding:1px 6px;border-radius:5px;white-space:nowrap}
  .tile.fp{border-color:var(--pred)} .tile.fn{border-color:var(--ref)} .tile.tp{border-color:rgba(66,224,122,.7)} .tile.dim{opacity:.18;background:#000}
  .vals{padding:6px 12px;border-top:1px solid var(--line);background:var(--panel);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .hint{margin-left:auto;color:var(--muted);font-size:11px} .hint b{color:var(--accent)}
</style></head><body>
<header>
  <h1>CMB QC <span class="a">__DATASET__</span></h1>
  <span class="sub">__N__ patients &middot; model __MODEL__ &middot; __LAYOUT__ &middot; slices at acquired spacing, scroll = z axis &middot; __GEN__</span>
  <div class="grp"><span class="lbl">brightness</span><input id="br" type="range" min="0.5" max="2.5" step="0.05" value="1">
    <span class="lbl">contrast</span><input id="ct" type="range" min="0.5" max="3" step="0.05" value="1"></div>
  <div class="grp"><button id="onlyFindings">only slices with findings</button><button id="nolist">hide list</button></div>
  <div class="grp"><button id="save" class="save">save verdicts</button><button id="pick">target folder</button><span id="target" class="sub"></span></div>
  <span id="overall" class="sub" style="flex-basis:100%;font-size:12.5px;color:var(--ink)"></span>
  <span class="hint"><b>&darr;/&uarr;</b> patient &middot; <b>r</b> reference &middot; <b>p</b> prediction &middot; <b>f</b> findings only &middot; <b>1-4</b> verdict &middot; <b>s</b> save</span>
</header>
<main>
  <aside><div class="tools"><input id="q" type="search" placeholder="filter patients ...">
    <div class="grp"><span class="lbl">sort</span><button data-k="id" class="on">name</button><button data-k="n_ref">lesions</button>
      <button data-k="f1" title="worst F1 first">F1</button><button data-k="fp">false</button><button data-k="fn">missed</button><button data-k="verdict">verdict</button></div></div>
    <div class="list" id="list"></div></aside>
  <section class="pane">
    <div class="phead"><span id="title" style="font-weight:700"></span><span id="counts" class="sub"></span>
      <span style="flex:1"></span>
      <span class="chip on" id="cRef"><i style="background:var(--ref)"></i>reference</span>
      <span class="chip on" id="cPred"><i style="background:var(--pred)"></i>prediction</span></div>
    <div id="scroll"><div class="colhead">__HEAD0__<div class="r">reference (ground truth)</div><div class="p">prediction (network)</div></div>
      <div id="sheet">__COL0__<div class="col"><img id="base"><img id="lRef" class="layer"><div id="tilesRef"></div></div>
        <div class="col"><img id="base2"><img id="lPred" class="layer"><div id="tilesPred"></div></div></div></div>
    <div class="vals"><span class="lbl">verdict</span><span id="verdicts" class="grp"></span>
      <input id="note" type="text" placeholder="note (optional)" style="flex:1;min-width:180px"></div>
  </section>
</main>
<script>
"use strict";
const DATASET = __DATASET_JSON__, MODEL = __MODEL_JSON__, DATA = __DATA__;
const KEY = 'cmb_qc_' + DATASET + '_' + MODEL, FILE = 'qc_' + DATASET + '_' + MODEL + '.json';
const VERDICTS = [['1','ok'],['2','image problem'],['3','annotation problem'],['4','prediction problem']];
let qc = JSON.parse(localStorage.getItem(KEY) || '{}'), state = {i:0, sort:'id', q:'', ref:true, pred:true, only:false}, folder = null;
const $ = id => document.getElementById(id);
// Lesion-level scores from the counts (26-connectivity, touch rule -- same definition as the training metric).
const fmt = v => v === null ? '&ndash;' : v.toFixed(2);
function scores(tp, fp, fn){ return { f1: 2*tp + fp + fn ? 2*tp / (2*tp + fp + fn) : null, sens: tp + fn ? tp / (tp + fn) : null, prec: tp + fp ? tp / (tp + fp) : null }; }
DATA.forEach(d => { if (d.has_prediction){ const s = scores(d.tp, d.fp, d.fn); d.f1 = s.f1; d.sens = s.sens; d.prec = s.prec; } });
function overall(){ const P = DATA.filter(d => d.has_prediction); if (!P.length) return 'no predictions in this page (reference only)';
  const tp = P.reduce((a, d) => a + d.tp, 0), fp = P.reduce((a, d) => a + d.fp, 0), fn = P.reduce((a, d) => a + d.fn, 0), s = scores(tp, fp, fn);
  const withLesions = P.filter(d => d.n_ref > 0).map(d => d.f1).sort((a, b) => a - b), m = withLesions.length;
  const median = m ? (m % 2 ? withLesions[(m - 1) / 2] : (withLesions[m / 2 - 1] + withLesions[m / 2]) / 2) : null;
  const clean = P.filter(d => d.n_ref === 0), cleanFp = clean.filter(d => d.fp > 0).length;
  return '<b>overall (' + P.length + ' patients, ' + (tp + fn) + ' lesions):</b> F1 <b>' + fmt(s.f1) + '</b> &middot; sensitivity <b>' + fmt(s.sens) + '</b> &middot; precision <b>' + fmt(s.prec) +
    '</b> &middot; <span class="tp">' + tp + ' found</span> <span class="fp">' + fp + ' false</span> <span class="fn">' + fn + ' missed</span> &middot; ' + (fp / P.length).toFixed(2) +
    ' false alarms per patient &middot; median patient F1 ' + fmt(median) + ' (' + m + ' patients with lesions) &middot; ' + cleanFp + ' of ' + clean.length + ' lesion-free patients have a false alarm'; }
function visible(){ let v = DATA.filter(d => d.id.toLowerCase().includes(state.q));
  const k = state.sort, val = d => k === 'verdict' ? ((qc[d.id]||{}).verdict || '') : (d[k] === undefined ? -1 : d[k]);
  v.sort((a, b) => k === 'id' || k === 'verdict' ? String(val(a)).localeCompare(String(val(b))) :
    k === 'f1' ? ((a.f1 === null || a.f1 === undefined ? 9 : a.f1) - (b.f1 === null || b.f1 === undefined ? 9 : b.f1)) : val(b) - val(a)); return v; }
function cur(){ const v = visible(); return v[Math.min(state.i, v.length - 1)]; }
function list(){ const el = $('list'), v = visible(); el.innerHTML = '';
  v.forEach((d, k) => { const r = document.createElement('div'), ver = (qc[d.id]||{}).verdict; r.className = 'row' + (k === Math.min(state.i, v.length - 1) ? ' on' : '');
    r.innerHTML = '<div class="sid">' + d.id + '<em>' + (ver ? (ver === 'ok' ? '<span class="tp">ok</span>' : '<span class="fp">' + ver + '</span>') : '') + '</em></div><div class="m">' + d.cohort +
      ' &middot; ' + d.n_ref + ' lesions' + (d.has_prediction ? ' &middot; <span class="tp">' + d.tp + ' found</span> <span class="fp">' + d.fp + ' false</span> <span class="fn">' + d.fn + ' missed</span><br>F1 <b>' + fmt(d.f1) + '</b> &middot; sens ' + fmt(d.sens) + ' &middot; prec ' + fmt(d.prec) : '') + '</div>';
    r.onclick = () => { state.i = k; show(); }; el.appendChild(r); });
  const on = el.querySelector('.on'); if (on) on.scrollIntoView({block:'nearest'}); }
function show(){ const d = cur(); list(); if (!d) return;
  $('title').textContent = d.id; $('counts').innerHTML = (d.has_prediction ? '<b style="color:var(--ink)">F1 ' + fmt(d.f1) + ' &middot; sensitivity ' + fmt(d.sens) + ' &middot; precision ' + fmt(d.prec) +
    '</b> &middot; <span class="tp">' + d.tp + ' found</span> <span class="fp">' + d.fp + ' false</span> <span class="fn">' + d.fn + ' missed</span> &middot; ' : '') + d.cohort + ' · fold ' + d.fold + ' · one tile = ' + d.step_mm + ' mm · ' + d.tiles.length + ' slices';
  ['base','base2','base0'].forEach(id => { const el = $(id); if (el) el.src = 'img/' + d.id + '.jpg'; }); $('lRef').src = 'img/' + d.id + '_ref.png'; $('lPred').src = 'img/' + d.id + '_pred.png';
  layers(); tiles(d); $('scroll').scrollTop = 0;
  const r = qc[d.id] || {}; document.querySelectorAll('#verdicts button').forEach(b => b.classList.toggle('on', b.dataset.v === r.verdict)); $('note').value = r.note || ''; }
function layers(){ $('lRef').style.display = state.ref ? 'block' : 'none'; $('lPred').style.display = state.pred ? 'block' : 'none';
  $('cRef').classList.toggle('on', state.ref); $('cPred').classList.toggle('on', state.pred);
  ['base','base2','base0'].forEach(id => $(id) && ($(id).style.filter = 'brightness(' + $('br').value + ') contrast(' + $('ct').value + ')')); $('onlyFindings').classList.toggle('on', state.only); }
function tiles(d){ const n = d.tiles.length;
  [['tilesRef','ref'],['tilesPred','pred']].forEach(([id, side]) => { const el = $(id); el.innerHTML = '';
    d.tiles.forEach((t, k) => { const box = document.createElement('div'), any = t.detected + t.missed + t.false > 0;
      const cls = side === 'ref' ? (t.missed ? ' fn' : t.detected ? ' tp' : '') : (t.false ? ' fp' : t.detected ? ' tp' : '');
      box.className = 'tile' + cls + (state.only && !any ? ' dim' : '');
      box.style.left = '0'; box.style.top = (100 * k / n) + '%'; box.style.width = '100%'; box.style.height = (100 / n) + '%';
      const text = side === 'ref' ? (t.detected + t.missed ? ' &middot; <span class="tp">' + t.detected + ' found</span> <span class="fn">' + t.missed + ' missed</span>' : '')
                                  : (t.detected + t.false ? ' &middot; <span class="tp">' + t.detected + ' correct</span> <span class="fp">' + t.false + ' false</span>' : '');
      box.innerHTML = '<span>z ' + t.z + text + '</span>'; el.appendChild(box); }); }); }
function persist(){ localStorage.setItem(KEY, JSON.stringify(qc)); }
function verdict(v){ const d = cur(); if (!d) return; const r = qc[d.id] = qc[d.id] || {}; r.verdict = r.verdict === v ? undefined : v; r.time = new Date().toISOString(); persist(); show(); }
async function save(){ const text = JSON.stringify({dataset:DATASET, model:MODEL, saved:new Date().toISOString(), patients:qc}, null, 1);
  if (folder){ try { const h = await folder.getFileHandle(FILE, {create:true}), w = await h.createWritable(); await w.write(text); await w.close(); $('target').textContent = 'saved ' + FILE + ' ' + new Date().toLocaleTimeString(); return; } catch(e){ alert('could not write: ' + e.message); } }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type:'application/json'})); a.download = FILE; a.click(); $('target').textContent = 'downloaded ' + FILE; }
VERDICTS.forEach(([k, v]) => { const b = document.createElement('button'); b.textContent = k + ' ' + v; b.dataset.v = v; b.onclick = () => verdict(v); $('verdicts').appendChild(b); });
$('save').onclick = save; $('pick').onclick = async () => { if (!window.showDirectoryPicker){ alert('This browser cannot pick a folder; saving downloads the file instead.'); return; } try { folder = await window.showDirectoryPicker({id:'cmb_qc', mode:'readwrite'}); $('target').textContent = 'target: ' + folder.name; } catch(e){} };
$('q').oninput = e => { state.q = e.target.value.toLowerCase(); state.i = 0; show(); };
document.querySelectorAll('[data-k]').forEach(b => b.onclick = () => { state.sort = b.dataset.k; state.i = 0; document.querySelectorAll('[data-k]').forEach(x => x.classList.toggle('on', x === b)); show(); });
$('cRef').onclick = () => { state.ref = !state.ref; layers(); }; $('cPred').onclick = () => { state.pred = !state.pred; layers(); };
$('onlyFindings').onclick = () => { state.only = !state.only; layers(); tiles(cur()); }; $('nolist').onclick = () => { const a = document.querySelector('aside'), m = document.querySelector('main'); const off = a.style.display === 'none'; a.style.display = off ? 'flex' : 'none'; m.style.gridTemplateColumns = off ? '270px 1fr' : '0 1fr'; };
$('br').oninput = layers; $('ct').oninput = layers; $('note').onchange = e => { const d = cur(); if (!d) return; (qc[d.id] = qc[d.id] || {}).note = e.target.value; persist(); };
window.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return; const n = visible().length, v = VERDICTS.find(x => x[0] === e.key);
  if (e.key === 'ArrowDown'){ state.i = Math.min(n - 1, state.i + 1); show(); } else if (e.key === 'ArrowUp'){ state.i = Math.max(0, state.i - 1); show(); }
  else if (e.key === 'r') $('cRef').click(); else if (e.key === 'p') $('cPred').click(); else if (e.key === 'f') $('onlyFindings').click(); else if (e.key === 's') save(); else if (v) verdict(v[1]); else return; e.preventDefault(); });
// Deep link: "qc.html#sub-107" opens that patient and scrolls to the first slice with a finding.
function openFromHash(){ const id = decodeURIComponent(location.hash.slice(1)); if (!id) return; const k = visible().findIndex(d => d.id === id); if (k < 0) return;
  state.i = k; show(); const d = cur(), first = d.tiles.findIndex(t => t.detected + t.missed + t.false > 0);
  $('base').onload = () => { if (first > 0) $('scroll').scrollTop = $('sheet').offsetHeight * first / d.tiles.length - 40; }; }
window.addEventListener('hashchange', openFromHash);
$('overall').innerHTML = overall();
show(); openFromHash();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--predictions", help="run folder with fold*/<id>_pred_proba.nii.gz (optional)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--columns", type=int, choices=(2, 3), default=2,
                        help="2 = reference | prediction (default, as built for VALDO on 2026-09-20); "
                             "3 = the plain image as a first column, so one can judge the slice without any outline drawn on it")
    parser.add_argument("--refresh-html", action="store_true",
                        help="only re-render qc.html from the data embedded in the existing page (layout / text changes)")
    args = parser.parse_args()
    absolute = lambda p: p if os.path.isabs(p) else f"{ROOT}/{p}"
    out = absolute(args.out); os.makedirs(f"{out}/img", exist_ok=True)
    predictions = absolute(args.predictions) if args.predictions else None
    cases = [e for m in args.manifest for e in json.load(open(absolute(m)))]
    if args.limit:
        cases = cases[:args.limit]
    data: List[Dict] = []
    if args.refresh_html:
        import re
        old = open(f"{out}/qc.html", encoding="utf-8").read()
        data = json.loads(re.search(r"DATA = (\[.*?\]);\nconst KEY", old, flags=re.S).group(1))
        cases = []
    for number, case in enumerate(cases, 1):
        data.append(build_patient(case, predictions, out))
        print(f"[{number}/{len(cases)}] {case['id']}: {len(data[-1]['tiles'])} tiles, one per {data[-1]['step_mm']} mm", flush=True)
    import time
    model = os.path.basename(predictions) if predictions else "none"
    drei = args.columns == 3
    page = (PAGE.replace("__DATASET_JSON__", json.dumps(args.dataset)).replace("__MODEL_JSON__", json.dumps(model))
                .replace("__DATA__", json.dumps(data)).replace("__DATASET__", args.dataset).replace("__MODEL__", model)
                .replace("__N__", str(len(data))).replace("__GEN__", time.strftime("%Y-%m-%d %H:%M"))
                .replace("__GRID__", "1fr 1fr 1fr" if drei else "1fr 1fr").replace("__MAXW__", "1800" if drei else "1300")
                .replace("__HEAD0__", '<div>image (no overlay)</div>' if drei else "")
                .replace("__COL0__", '<div class="col"><img id="base0"></div>' if drei else "")
                .replace("__LAYOUT__", "image / reference / prediction" if drei else "left: reference, right: prediction"))
    open(f"{out}/qc.html", "w", encoding="utf-8").write(page)
    print(f"wrote {out}/qc.html")


if __name__ == "__main__":
    main()

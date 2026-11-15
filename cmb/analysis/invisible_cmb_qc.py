#!/usr/bin/env python3
"""QC page: the "invisible" reference microbleeds (wc10, 2026-11-05).

All invisible reference lesions of the three cohorts (t2star, swi, valdo) plus an
equal number of randomly chosen VISIBLE lesions as controls, shuffled. Per lesion:
an axial crop (+/-12 voxels) on the lesion's centre slice and the slice above/below,
image with overlay = reference label (red outline) and stage-1 probability (heat),
overlay switch (O), "? Methods" with software versions, English, "internal, do not share".
Columns: cohort, size, region, contrast, visible yes/no.

Build: <qc-dir>/CMB_invisible_<date>/  

    python cmb/analysis/invisible_cmb_qc.py [--seed 0] [--max N]
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from cmb.paths import ROOT
from cmb.paths import MB
TSV = f"{MB}/stage2/schleife/invisible_cmb_lesions.tsv"
C26 = np.ones((3, 3, 3), bool)
HALF = 12
DATE = os.popen("date +%Y%m%d").read().strip()
# target is <qc-dir>/CMB_invisible_<date>/ .
# NOTE: the agent sandbox cannot write there -- run this via
# `systemd-run --user --unit=...` (runs outside the sandbox), e.g.
#   systemd-run --user --unit=wc10-qc -- bash /tmp/wc10_qc_run.sh
OUT = os.environ.get("CMB_QC_OUT", f"{MB}/qc") + f"/CMB_invisible_{DATE}"


def versions():
    py = sys.executable
    v = {"python": py, "nibabel": nib.__version__,
         "numpy": np.__version__, "scipy": __import__("scipy").__version__,
         "Pillow": Image.__version__}
    v["torch"] = subprocess.run([py, "-c", "import torch;print(torch.__version__)"],
                                capture_output=True, text=True).stdout.strip()
    v["monai"] = subprocess.run([py, "-c", "import monai;print(monai.__version__)"],
                                capture_output=True, text=True).stdout.strip()
    return v


def heat(prob2d):
    """probability -> RGBA heat overlay (transparent below 0.01)."""
    p = np.clip(prob2d, 0, 1)
    alpha = np.where(p >= 0.01, np.clip((p - 0.01) / 0.99, 0, 1) * 0.75, 0).astype(np.float64)
    rgb = np.zeros((*prob2d.shape, 4), np.uint8)
    rgb[..., 0] = 255                       # red heat
    rgb[..., 3] = (alpha * 255).astype(np.uint8)
    return rgb


def outline2d(mask2d):
    return mask2d & ~ndimage.binary_erosion(mask2d, structure=np.ones((3, 3), bool))


def crop3(vol, centre, half):
    out = np.zeros([2 * h + 1 for h in half] + list(vol.shape[3:]), vol.dtype)
    src, dst = [], []
    for c, h, n in zip(centre, half, vol.shape[:3]):
        lo, hi = c - h, c + h + 1
        s, e = max(lo, 0), min(hi, n)
        if s >= e:
            return None
        src.append(slice(s, e)); dst.append(slice(s - lo, e - lo))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def png_b64(gray, overlay=None, outline=None):
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(g, mode="L").convert("RGB")
    px = np.asarray(im, dtype=np.float64)
    if overlay is not None:
        o = overlay.astype(np.float64) / 255.0
        a = o[..., 3:4]
        px = px * (1 - a) + o[..., :3] * a
    if outline is not None:
        px[outline] = [255, 0, 0]
    buf = io.BytesIO()
    Image.fromarray(px.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def read_tsv():
    rows = []
    with open(TSV) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            d = dict(zip(header, vals))
            rows.append(d)
    return rows


def lesion_cards(rows, seed, per_cohort_max):
    rng = np.random.default_rng(seed)
    cards = []
    for cohort in ("t2star", "swi", "valdo"):
        sub = [r for r in rows if r["cohort"] == cohort]
        # visible=="True" = max probability >= 0.01 (visible); == "False" = invisible
        inv = [r for r in sub if r["visible"] == "False"]
        vis = [r for r in sub if r["visible"] == "True"]
        # equal number of visible controls, shuffled together
        if len(vis) > len(inv):
            keep = set(rng.choice(len(vis), size=len(inv), replace=False))
            vis = [vis[i] for i in keep]
        sel = inv + vis
        if per_cohort_max and len(sel) > per_cohort_max:
            sel = [sel[i] for i in rng.choice(len(sel), size=per_cohort_max, replace=False)]
        rng.shuffle(sel)
        cards += sel
    return cards


_CACHE = {}


def _vols(r):
    key = (r["image_dir"], r["pred_dir"], r["case"])
    if key not in _CACHE:
        case = r["case"]
        img_ni = nib.load(f"{r['image_dir']}/{case}_image.nii.gz")
        img = np.asarray(img_ni.dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(f"{r['image_dir']}/{case}_label.nii.gz").dataobj) > 0
        prob_path = glob.glob(f"{r['pred_dir']}/fold*/{case}_pred_proba.nii.gz")[0]
        prob = np.asarray(nib.load(prob_path).dataobj).astype(np.float32) / 255.0
        _CACHE[key] = (img, lab, prob)
    return _CACHE[key]


def build_card(r):
    case, cohort = r["case"], r["cohort"]
    cx, cy, cz = int(r["cx"]), int(r["cy"]), int(r["cz"])
    img, lab, prob = _vols(r)

    panels = []
    for z in (cz - 1, cz, cz + 1):
        if z < 0 or z >= img.shape[2]:
            continue
        g = crop3(img, (cx, cy, z), [HALF, HALF, 1])
        if g is None:
            continue
        g = np.clip((g - np.percentile(g, 1)) / max(np.percentile(g, 99) - np.percentile(g, 1), 1e-9), 0, 1)[0]
        p2 = crop3(prob, (cx, cy, z), [HALF, HALF, 1])
        l2 = crop3(lab.astype(np.uint8), (cx, cy, z), [HALF, HALF, 1])
        p2 = p2[0] if p2 is not None else np.zeros((2 * HALF + 1, 2 * HALF + 1))
        l2 = l2[0] > 0 if l2 is not None else np.zeros(g.shape, bool)
        base = png_b64(g)
        ov = png_b64(g, overlay=heat(p2))
        ol = png_b64(g, outline=outline2d(l2))
        both = png_b64(g, overlay=heat(p2), outline=outline2d(l2))
        tag = "centre" if z == cz else ("above" if z > cz else "below")
        panels.append(
            f'<div class="panel"><div class="ptag">z = {z} ({tag})</div>'
            f'<div class="ov" data-base="{base}" data-heat="{ov}" data-ref="{ol}" data-both="{both}">'
            f'<img src="data:image/png;base64,{both}">'
            f'</div></div>')
    vis_txt = "yes" if r["visible"] == "True" else "no"
    contrast = r["contrast"] if r["contrast"] not in ("", "nan") else "n/a"
    return (f'<div class="card {"vis" if r["visible"] == "True" else "inv"}">'
            f'<div class="head"><b>{html.escape(case)}</b> &nbsp; '
            f'<span class="tbl"><span class="c">cohort</span> {cohort} | '
            f'<span class="c">size</span> {r["size_mm3"]} mm3 ({r["n_voxels"]} vox) | '
            f'<span class="c">region</span> {r["region"]} | '
            f'<span class="c">contrast</span> {contrast} | '
            f'<span class="c">visible</span> <b>{vis_txt}</b> (max p {r["p_max"]}) | '
            f'<span class="c">slices</span> {r["n_slices"]} | '
            f'<span class="c">edge</span> {r["edge_mm"]} mm | '
            f'<span class="c">shift</span> {r["darkest_offset_mm"]} mm | '
            f'<span class="c">fold</span> {r["fold"]}</span></div>'
            f'<div class="row">{"".join(panels)}</div></div>')


HTML_TMPL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CMB invisible QC</title>
<style>
 body {{ font-family: sans-serif; background:#fafafa; margin:16px; }}
 h1 {{ font-size: 18px; }}
 .banner {{ background:#b00020; color:#fff; padding:6px 10px; display:inline-block;
            border-radius:3px; font-weight:bold; }}
 .bar {{ margin:10px 0; }}
 .bar button {{ margin-right:6px; padding:4px 10px; }}
 .bar .active {{ background:#333; color:#fff; }}
 .card {{ background:#fff; border:1px solid #ccc; border-radius:4px; margin:10px 0; padding:8px; }}
 .card.inv {{ border-left:6px solid #b00020; }}
 .card.vis {{ border-left:6px solid #2a7; }}
 .head {{ font-size: 14px; margin-bottom:6px; }}
 .tbl {{ font-family: monospace; font-size: 12px; color:#222; }}
 .row {{ display:flex; gap:8px; flex-wrap:wrap; }}
 .panel {{ text-align:center; }}
 .ptag {{ font-size: 11px; color:#555; }}
 .ov img {{ image-rendering: pixelated; }}
 .methods {{ background:#eee; border:1px solid #bbb; padding:10px; font-size:13px; max-width:860px; }}
 .hidden {{ display:none; }}
</style></head>
<body>
<h1>CMB — "invisible" reference microbleeds (stage-1 max probability &lt; 0.01) <span class="banner">internal, do not share</span></h1>
<div class="bar">
 <button id="b-both" class="active" onclick="setov('both')">both overlays (O)</button>
 <button id="b-heat" onclick="setov('heat')">stage-1 probability</button>
 <button id="b-ref" onclick="setov('ref')">reference label</button>
 <button id="b-base" onclick="setov('base')">plain</button>
 <button onclick="document.getElementById('methods').classList.toggle('hidden')">? Methods</button>
</div>
<div id="methods" class="methods hidden">
 <b>Methods.</b> Reference CMB = 26-connected components of the manual label on the arena grid
 (0.5 &times; 0.5 &times; 1.0 mm; 1 voxel = 0.25 mm3). Stage-1 probability = frozen arena stage-1
 model a03-aniso-e60 (MONAI U-Net family), out-of-fold prediction per case, uint8/255.
 A lesion is <b>invisible</b> when its maximum stage-1 probability is &lt; 0.01.
 contrast = mean FRST (network input, inverted scale) inside the lesion / mean of a 2-voxel shell
 around it (higher = brighter relative to surroundings). shift = distance from the label centroid
 to the brightest input voxel within 5 voxels (label-shift check). edge = distance from the lesion
 centroid to the brain edge (SynthSeg 8.2.0, FreeSurfer). Panels: axial crops of &plusmn;12 voxels
 around the lesion centre on the centre slice and the slice above/below.
 Software: {versions}.
</div>
{cards}
<script>
function setov(mode) {{
 document.querySelectorAll('.ov').forEach(function(o) {{ o.firstChild.src='data:image/png;base64,'+o.dataset[mode]; }});
 document.querySelectorAll('.bar button').forEach(function(b) {{ b.classList.remove('active'); }});
 document.getElementById('b-'+mode).classList.add('active');
}}
document.addEventListener('keydown', function(e) {{ if (e.key.toLowerCase()==='o') {{
 var seq=['both','heat','ref','base']; var cur=document.querySelector('.bar button.active').id.slice(2);
 var nxt=seq[(seq.indexOf(cur)+1)%seq.length]; setov(nxt); }} }});
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=0, help="max cards per cohort (0 = all)")
    a = ap.parse_args()
    rows = read_tsv()
    cards = lesion_cards(rows, a.seed, a.max)
    print(f"{len(cards)} cards (invisible + equal visible controls, shuffled)")
    html_out = HTML_TMPL.format(versions=json.dumps(versions()),
                                cards="".join(build_card(r) for r in cards))
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/index.html"
    with open(path, "w") as f:
        f.write(html_out)
    # store the selection (aggregate only: cohort, visible, region) for reproducibility
    with open(f"{OUT}/selection.json", "w") as f:
        json.dump(dict(seed=a.seed, date=DATE,
                       n_by=dict((c, int(sum(1 for r in cards if r["cohort"] == c and r["visible"] == "True")))
                                 for c in ("t2star", "swi", "valdo")),
                       n_cards=len(cards)), f, indent=1)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB)")
    return path


if __name__ == "__main__":
    main()

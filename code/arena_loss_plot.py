#!/usr/bin/env python3
"""Loss and inner-fold F1 per epoch for the arena runs, as an inline SVG block for the public architecture page.

Reads ``ergebnisse/turnier/<run>/fold*/verlauf.json`` (training loss every epoch, inner-fold lesion F1 every five
epochs), averages over the folds that reached the epoch, and writes one ``<details>`` block per round with two panels
(loss, inner F1), a legend, a crosshair tooltip and a table view. Output: ``seiten/arena_loss_fragment.html``;
``--insert PAGE`` replaces the markers ``<!-- arena-loss-round1 -->`` / ``<!-- arena-loss-round2 -->`` in PAGE.

Colours: the eight validated categorical slots of the dataviz palette in fixed order (winner first); a ninth series is grey.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

ROOT = "/home/uchralt/data/work/valdo-t2s"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8f98"

ROUND1 = [  # run folder, label (as in the page table), colour slot
    ("a03-aniso", "anisotropic 3D U-Net (winner)"),
    ("a04-2p5d", "2.5D U-Net"),
    ("a02-nnunet", "plain 3D U-Net, nnU-Net style"),
    ("a06-unetpp", "UNet++"),
    ("a01-attunet", "attention U-Net (reference)"),
    ("a07-flach", "shallow dilated U-Net"),
    ("a05-resse", "residual SE U-Net"),
    ("a09-segres", "SegResNet"),
    ("a08-swin", "Swin UNETR (1 fold)"),
]
ROUND3 = [  # why a fixed schedule: same network, 60 / 90 / 120 epochs, plus the round-1 recipe with early stopping
    ("a03-aniso-e60-gd", "60 epochs, fixed (the recipe)"),
    ("a03-aniso-e90-gd", "90 epochs, fixed"),
    ("a03-aniso-e120-gd", "120 epochs, fixed"),
    ("a03-aniso-s1", "round-1 recipe: early stopping, second seed"),
]
ROUND2 = [
    ("a03-aniso-e60-gd", "anisotropic 3D U-Net, 32 channels (baseline)"),
    ("a11-anisoatt-e60-gd", "+ attention gates"),
    ("a03-aniso-b16-e60-gd", "16 base channels"),
    ("a03-aniso-b48-e60-gd", "48 base channels"),
    ("a13-r2unet2p5d-e60-gd", "R2U-Net 2.5D"),
    ("a14-mednext-e60-gd", "MedNeXt"),
    ("a15-nnunet-resenc-e60-gd", "nnU-Net residual encoder"),
]


def series(run: str) -> dict:
    per_loss, per_f1, folds = {}, {}, 0
    for f in sorted(glob.glob(f"{ROOT}/ergebnisse/turnier/{run}/fold*/verlauf.json")):
        folds += 1
        for row in json.load(open(f)):
            per_loss.setdefault(row["epoche"], []).append(row["loss"])
            if "innen_f1" in row:
                per_f1.setdefault(row["epoche"], []).append(row["innen_f1"])
    loss = [(e, round(st.mean(v), 4), len(v)) for e, v in sorted(per_loss.items())]
    f1 = [(e, round(st.mean(v), 4), len(v)) for e, v in sorted(per_f1.items())]
    return dict(folds=folds, loss=loss, f1=f1)


def panel(data, key, y_max, y_label, width=440, height=250, x_max=60):
    """One SVG panel: lines (2px, round), hairline grid, end markers for the first two series."""
    ml, mr, mt, mb = 44, 14, 14, 30
    pw, ph = width - ml - mr, height - mt - mb
    sx = lambda e: ml + (e - 1) / (x_max - 1) * pw
    sy = lambda v: mt + (1 - v / y_max) * ph
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{y_label} per epoch">']
    steps = 5
    for i in range(steps + 1):
        v = y_max * i / steps
        y = sy(v)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width - mr}" y2="{y:.1f}" stroke="#e3e7ec" stroke-width="1"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5b6878">{v:.2f}</text>')
    for e in range(0, x_max + 1, 10 if x_max > 40 else 5):
        if e == 0:
            continue
        out.append(f'<text x="{sx(e):.1f}" y="{height - 10}" text-anchor="middle" font-size="11" fill="#5b6878">{e}</text>')
    out.append(f'<text x="{width / 2:.0f}" y="{height - 1}" text-anchor="middle" font-size="11" fill="#5b6878">epoch</text>')
    out.append(f'<text x="12" y="{mt + ph / 2:.0f}" transform="rotate(-90 12 {mt + ph / 2:.0f})" text-anchor="middle" font-size="11" fill="#5b6878">{y_label}</text>')
    for idx, (run, label, colour, d) in enumerate(data):
        pts = [(sx(e), sy(min(v, y_max))) for e, v, n in d[key]]
        if not pts:
            continue
        path = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="{1 if idx < 2 else 0.85}"/>')
        if idx < 2:
            x, y = pts[-1]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#fff"/><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>')
    out.append(f'<line class="xh" x1="0" y1="{mt}" x2="0" y2="{mt + ph}" stroke="#1c2430" stroke-width="1" opacity="0"/>')
    out.append("</svg>")
    return "\n".join(out), dict(ml=ml, pw=pw, x_max=x_max)


def block(name, runs, x_max, title, note):
    data = []
    for i, (run, label) in enumerate(runs):
        d = series(run)
        data.append((run, label, PALETTE[i] if i < len(PALETTE) else GREY, d))
    loss_svg, geo = panel(data, "loss", 0.75, "training loss (BCE + Dice)", x_max=x_max)
    f1_svg, _ = panel(data, "f1", 0.7, "inner-fold lesion F1", x_max=x_max)
    legend = "".join(f'<span class="k"><i style="background:{c}"></i>{label}</span>' for _, label, c, _ in data)
    rows = "".join(
        f"<tr><td><i class='sw' style='background:{c}'></i>{label}</td><td class='r'>{d['folds']}</td><td class='r'>{d['loss'][-1][0]}</td>"
        f"<td class='r'>{d['loss'][0][1]:.3f}</td><td class='r'>{min(v for _, v, _ in d['loss']):.3f}</td><td class='r'>{d['loss'][-1][1]:.3f}</td>"
        f"<td class='r'>{(max(v for _, v, _ in d['f1']) if d['f1'] else float('nan')):.3f}</td></tr>" for _, label, c, d in data)
    payload = json.dumps([dict(label=label, colour=c, loss=[[e, v] for e, v, _ in d["loss"]], f1=[[e, v] for e, v, _ in d["f1"]]) for _, label, c, d in data])
    return f'''<details class="step" data-s="tour" id="{name}"><summary><span class="n">&#8599;</span><span class="t">{title}</span><span class="shape">tap to open</span></summary>
<div class="viz" data-geo='{json.dumps(geo)}' data-series='{payload}'>
<p>{note}</p>
<div class="panels"><div class="panel" data-key="loss">{loss_svg}</div><div class="panel" data-key="f1">{f1_svg}</div></div>
<div class="legend-lines">{legend}</div>
<div class="tip" hidden></div>
<details class="tbl"><summary>table view</summary><div class="scroll"><table>
<tr><th>Network</th><th class="r">folds</th><th class="r">epochs</th><th class="r">loss, epoch 1</th><th class="r">loss, minimum</th><th class="r">loss, last</th><th class="r">inner F1, best</th></tr>{rows}</table></div></details>
</div></details>'''


STYLE = '''<style>
  .viz { margin:.5rem 0 .2rem 0 !important; font-size:.9rem; }
  .viz .panels { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; } @media (max-width:700px){ .viz .panels { grid-template-columns:1fr; } }
  .viz .panel { position:relative; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.3rem; }
  .viz .legend-lines { display:flex; flex-wrap:wrap; gap:.3rem 1rem; margin:.5rem 0; font-size:.85rem; color:var(--ink); }
  .viz .k i, .viz .sw { display:inline-block; width:18px; height:3px; border-radius:2px; vertical-align:middle; margin-right:.4rem; }
  .viz .tip { position:fixed; z-index:9; background:#fff; border:1px solid var(--line); border-radius:8px; padding:.4rem .6rem; font-size:.8rem; box-shadow:0 2px 8px rgba(0,0,0,.12); pointer-events:none; max-width:300px; }
  .viz .tip b { display:block; margin-bottom:.2rem; } .viz .tip .row { display:flex; gap:.5rem; align-items:center; } .viz .tip .row i { width:14px; height:3px; display:inline-block; border-radius:2px; }
  .viz .tip .row strong { font:600 .85rem ui-monospace,Menlo,monospace; min-width:3.2em; }
  details.tbl summary { cursor:pointer; color:var(--soft); font-size:.85rem; } details.tbl td.r, details.tbl th.r { text-align:right; }
</style>'''

SCRIPT = '''<script>
(function(){
  document.querySelectorAll('.viz').forEach(function(viz){
    var geo = JSON.parse(viz.dataset.geo), S = JSON.parse(viz.dataset.series), tip = viz.querySelector('.tip');
    viz.querySelectorAll('.panel').forEach(function(panel){
      var svg = panel.querySelector('svg'), key = panel.dataset.key, xh = svg.querySelector('.xh');
      function show(ev){
        var r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal, x = (ev.clientX - r.left) * vb.width / r.width;
        var e = Math.round((x - geo.ml) / geo.pw * (geo.x_max - 1)) + 1; if (e < 1 || e > geo.x_max) { hide(); return; }
        var sx = geo.ml + (e - 1) / (geo.x_max - 1) * geo.pw; xh.setAttribute('x1', sx); xh.setAttribute('x2', sx); xh.setAttribute('opacity', '0.5');
        tip.textContent = ''; var h = document.createElement('b'); h.textContent = (key === 'loss' ? 'training loss, epoch ' : 'inner-fold F1, epoch ') + e; tip.appendChild(h);
        S.forEach(function(s){ var p = s[key].filter(function(q){ return q[0] === e; })[0]; if (!p) return;
          var row = document.createElement('div'); row.className = 'row'; var i = document.createElement('i'); i.style.background = s.colour;
          var v = document.createElement('strong'); v.textContent = p[1].toFixed(3); var l = document.createElement('span'); l.textContent = s.label;
          row.appendChild(i); row.appendChild(v); row.appendChild(l); tip.appendChild(row); });
        tip.hidden = false; var tw = tip.offsetWidth, th = tip.offsetHeight;
        tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tw - 8) + 'px'; tip.style.top = Math.max(8, Math.min(ev.clientY - th / 2, window.innerHeight - th - 8)) + 'px';
      }
      function hide(){ tip.hidden = true; xh.setAttribute('opacity', '0'); }
      svg.addEventListener('pointermove', show); svg.addEventListener('pointerleave', hide);
    });
  });
})();
</script>'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--insert", help="page to patch in place (markers arena-loss-round1 / -round2 and arena-loss-assets)")
    a = ap.parse_args()
    b1 = block("loss-round1", ROUND1, 40,
               "Round 1 as it happened: training loss and inner-fold F1 per epoch",
               "Mean over the folds that reached the epoch (early stopping ended some folds before epoch 40; Swin UNETR finished one fold). "
               "The loss falls for every network, but the loss is no judge: the shallow dilated U-Net reaches the same training loss as the winner (0.154) and the residual SE U-Net nearly so (0.155), "
               "yet they score 0.176 and 0.105 on the test folds against 0.404 (a voxel loss is cheap to lower with many small false alarms). The F1 on the inner fold is what chose the winner and the checkpoint. Hover for the values of every network at one epoch.")
    b2 = block("loss-round2", ROUND2, 60,
               "Round 2, architectures: training loss and inner-fold F1 per epoch (60 epochs, final recipe)",
               "Same loss function in every run (BCE + Dice with deep supervision), so the curves are comparable. MedNeXt and R2U-Net start lower and end higher; "
               "16 base channels reach the baseline's loss but not its F1. The recipe variants of round 2 (learning rate, batch, patches, lesion share, focal Tversky) are not drawn: a different loss or budget per epoch would not be comparable on this axis.")
    b3 = block("loss-round3", ROUND3, 120,
               "Why 60 fixed epochs and not early stopping: loss and inner-fold F1 at 60 / 90 / 120 epochs",
               "Round 1 trained with early stopping on the inner-fold F1 (checked every 5 epochs, patience 3 checks, at most 40 epochs). "
               "That F1 scatters by about 0.08 between random seeds, so the stop fires on noise: with the second seed one fold kept the checkpoint of epoch 5 and stopped at epoch 20, and the run scored 0.321 instead of 0.404 "
               "(lab book 4.5). A fixed cosine schedule of 60 epochs took the recipe to 0.470 and removed the seed dependence of the length. "
               "Longer is not better: at 90 and 120 epochs the training loss keeps falling and the best inner-fold value creeps up (it is a maximum over more checks), but the out-of-fold score does not follow; the paired result is -0.016 [-0.091; +0.060] for 90 and "
               "-0.029 on VALDO / -0.022 confirmed on the in-house T2* cohort for 120 (lab book 5h, round 2). The checkpoint is still chosen on the inner fold; taking the last state instead "
               "costs -0.023 [-0.069; +0.012], within noise (lab book 5o), which is what the full-data models of the product do.")
    frag = STYLE + "\n" + b1 + "\n" + b2 + "\n" + b3 + "\n" + SCRIPT
    os.makedirs(f"{ROOT}/seiten", exist_ok=True)
    open(f"{ROOT}/seiten/arena_loss_fragment.html", "w").write(frag)
    print(f"fragment written: {len(frag)} bytes")
    if a.insert:
        s = open(a.insert, encoding="utf-8").read()
        for marker, html in (("<!-- arena-loss-round1 -->", b1), ("<!-- arena-loss-round2 -->", b2 + "\n" + b3), ("<!-- arena-loss-assets -->", STYLE + "\n" + SCRIPT)):
            assert marker in s, marker
            s = s.replace(marker, marker + "\n" + html)
        open(a.insert, "w", encoding="utf-8").write(s)
        print(f"inserted into {a.insert}")


if __name__ == "__main__":
    main()

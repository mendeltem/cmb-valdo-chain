"""Render Table S2 (arena catalogue, from the manuscript draft table) and Table S3 (arena results, rules v2) as HTML for PNG."""
import os as _os, sys as _sys  # paper version: paths from the environment (see cmb/paths.py, PAPER_VERSION.md)
_R = _os.environ.get("CMB_ROOT", _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
_P = _os.environ.get("CMB_PRIVATE", _R + "/private")
_VALDO = _os.environ.get("CMB_VALDO", _R + "/data/valdo2021/Task2")
_MOMENI = _os.environ.get("CMB_MOMENI", _R + "/data/momeni-csiro-cmb")
_CASES = _os.environ.get("CMB_CASES", _P + "/faelle.json")
_PY = _os.environ.get("CMB_PYTHON", _sys.executable)
_EXTMODELS = _os.environ.get("CMB_EXTERN_MODELS", _R + "/models/extern")
_LABEL = _os.environ.get("CMB_INHOUSE_LABEL", "inhouse")
import json, os, re, html, sys
P = _R + "/paper"; M = _P + "/stage2"; R = sys.argv[1]
CSS = """body{background:#fff;margin:0;padding:24px 28px;font-family:Helvetica,Arial,sans-serif;color:#111}
h1{font-size:18px;margin:0 0 12px}table{border-collapse:collapse;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}
thead th{border-top:2px solid #111;border-bottom:2px solid #111}tbody tr:last-child td{border-bottom:2px solid #111}
td.n{text-align:right;white-space:nowrap}tr.ctl td{background:#f3f3f3}.notes{font-size:12px;color:#444;margin-top:10px;line-height:1.45}
.pos{color:#2e7d32}.neg{color:#b3261e}"""
def page(fn, title, head, rows, notes, width):
    h = "".join(f"<th>{x}</th>" for x in head)
    open(f"{R}/{fn}.html", "w").write(f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS} table,.notes{{width:{width}px}}</style></head><body><h1>{title}</h1><table><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table><div class="notes">{notes}</div></body></html>')
md = lambda s: html.escape(s).replace("\\*", "*")
# S1
t = open(f"{P}/manuscript.md").read()
blk = t[t.index("### Table S2 (draft)"):t.index("### Open checks")]
lines = [l for l in blk.splitlines() if l.startswith("|")][2:]
rows = "".join("<tr>" + "".join(f"<td>{md(c.strip())}</td>" for c in l.strip("|").split("|")) + "</tr>" for l in lines)
page("tableS1_arena_catalog", "Table S2. Architecture families considered for the component arena", ["Family", "Method (reference)", "Input", "Code", "Use in this study"], rows,
     "All references verified against Crossref and PubMed. Excluded: Lu et al. (2021), retracted (PMID 38213483). Repositories without a verified paper are not cited.", 1300)
# S2
fin = json.load(open(f"{M}/tset_summary_final.json"))["results"]; uni = json.load(open(f"{M}/tset_summary_union_final.json"))["results"]
ctl = json.load(open(f"{M}/tset_tset-plain_s0.json"))["results"]["pooled tset-plain"]
NAMES = {"pooled tset-ctx": "soft tissue context", "pooled tset-d4": "D4 group convolutions", "pooled tset-d4proto": "D4 + prototype loss",
         "pooled tset-full": "all modules (full)", "pooled tset-proto": "prototype margin loss", "pooled tset-set": "set attention over candidates",
         "pooled tset-track": "slice track + transformer", "pooled tset-tta8": "test-time augmentation (8 views)", "pooled tset-wide-tta8": "wide CNN + TTA (compute control)",
         "pooled tset-plain": "plain mixed 3D CNN", "probability": "stage 1 only (threshold)", "probability+csf": "stage 1 only + CSF rule"}
f = lambda x: f'<span class="{"pos" if x > 0 else "neg"}">{x:+.3f}</span>'
def row(name, r, stage1="base"):
    c = r["delta_cohorts"]
    return (f'<tr><td>{stage1}</td><td>{name}</td><td class="n">{",".join(map(str, r["seeds"]))}</td><td class="n">{f(r["delta_all"])} ({r["shrunk_delta"]:+.3f})</td>'
            f'<td class="n">{f(c["valdo"])}</td><td class="n">{f(c["t2star"])}</td><td class="n">{f(c["swi"])}</td><td class="n">{r["simultaneous_lower"]:+.3f}</td><td>{r["decision"]}</td></tr>')
rows = (f'<tr class="ctl"><td>base</td><td><b>control: plain mixed 3D CNN</b></td><td class="n">0,1</td><td class="n">F1 (seed 0) =</td><td class="n">{ctl["valdo"]["f1"]:.3f}</td><td class="n">{ctl["t2star"]["f1"]:.3f}</td><td class="n">{ctl["swi"]["f1"]:.3f}</td><td></td><td>standard</td></tr>')
order = sorted((k for k in fin if k in NAMES and not k.startswith("probability")), key=lambda k: -fin[k]["shrunk_delta"])
rows += "".join(row(NAMES[k], fin[k]) for k in order)
rows += "".join(row(NAMES.get(k.replace(" union", ""), k) + (" + vote count" if "ctx" in k else ""), v, "merged (sensitive)") for k, v in uni.items() if k.endswith("union") and "tset" in k)
rows += "".join(row(NAMES[k], fin[k]) for k in ("probability", "probability+csf") if k in fin)
page("tableS2_arena_results", "Table S3. Component arena: stage-2 modules and stage-1 candidate sets against the plain mixed classifier",
     ["Stage 1", "Stage 2 arm", "Seeds", "ΔF1 all (shrunk)", "ΔF1 VALDO", "ΔF1 T2*GRE", "ΔF1 SWI", "Simultaneous lower bound", "Decision"], rows,
     "ΔF1 = lesion-level F1 minus the control (plain mixed 3D CNN, seeds 0 and 1; first row gives its absolute F1 per dataset). Decision rules fixed before the runs (v2): screening with one seed advances if ΔF1(all) ≥ +0.01 and no dataset ≤ −0.02; selection with ≥ 2 seeds carries to confirmation only if the max-T simultaneous lower bound is > 0. Shrunk = estimate corrected for selection of the best arm. Merged stage 1 = base U-Net ∪ blob-loss U-Net. No arm was carried to confirmation.", 1300)
print("ok")

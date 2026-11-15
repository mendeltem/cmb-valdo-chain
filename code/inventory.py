import nibabel as nib, numpy as np, glob, os, json, collections, sys
from scipy import ndimage
S = ndimage.generate_binary_structure(3,3)   # 26er-Nachbarschaft
# Path to the unpacked VALDO Task 2 directory (the one holding sub-*/).
if len(sys.argv) < 2:
    sys.exit("usage: inventory.py <path to VALDO Task2>  (the directory holding sub-*/)")
root = sys.argv[1]
zeilen=[]
for d in sorted(glob.glob(f"{root}/sub-*")):
    sid=os.path.basename(d)
    cmb=nib.load(f"{d}/{sid}_space-T2S_CMB.nii.gz")
    im=nib.load(f"{d}/{sid}_space-T2S_desc-masked_T2S.nii.gz")
    lab,n=ndimage.label(cmb.get_fdata()>0, structure=S)
    z=np.round(im.header.get_zooms()[:3],2)
    d2=im.get_fdata()
    zeilen.append(dict(id=sid, kohorte=sid[4], n_cmb=int(n), shape=list(im.shape),
                       zooms=[float(x) for x in z], hirnanteil=round(float((d2>0).mean()),3),
                       t1=os.path.exists(f"{d}/{sid}_space-T2S_desc-masked_T1.nii.gz"),
                       t2=os.path.exists(f"{d}/{sid}_space-T2S_desc-masked_T2.nii.gz")))
json.dump(zeilen, open("valdo_inventar.json","w"), indent=1)

def klasse(n): return "0" if n==0 else ("1-2" if n<=2 else ("3-5" if n<=5 else ">5"))
print("Faelle:", len(zeilen), " CMB gesamt:", sum(z["n_cmb"] for z in zeilen))
print("\nZellen  Kohorte x Lastklasse (4 Klassen):")
t=collections.Counter((z["kohorte"],klasse(z["n_cmb"])) for z in zeilen)
kl=["0","1-2","3-5",">5"]
print("  koh |" + "".join(f"{k:>6}" for k in kl) + "   ges")
for k in "123":
    print(f"    {k} |" + "".join(f"{t[(k,c)]:>6}" for c in kl) + f"{sum(t[(k,c)] for c in kl):>6}")
print("\nGeometrie je Kohorte:")
for k in "123":
    g=[z for z in zeilen if z["kohorte"]==k]
    zs=collections.Counter(tuple(z["zooms"]) for z in g)
    sh=collections.Counter(tuple(z["shape"]) for z in g)
    print(f"  {k}: n={len(g)}  CMB={sum(z['n_cmb'] for z in g)}  T1={sum(z['t1'] for z in g)}  T2={sum(z['t2'] for z in g)}")
    for v,c in zs.most_common(3): print(f"       pixdim {v} x{c}  -> Anisotropie {max(v)/min(v):.1f}:1")
    for v,c in sh.most_common(3): print(f"       shape  {v} x{c}")

"""Predict microbleeds on one scan with the public VALDO T2* weights (scan and output folder: CMB_SCAN, CMB_OUTPUT_DIR
in .env; the shipped scan is a synthetic phantom -- point CMB_SCAN at your own T2*-GRE scan)."""

from cmb import env, get_model, predict

get_model("valdo_t2")  # downloaded once, SHA-256 checked
result = predict(env())
print(f"{result['n']} CMB found -> {result['csv']}")

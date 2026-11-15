"""Fine-tune the public VALDO T2* U-Net on local scans with the paper's Phase 2 protocol (full fine-tuning, lr 1e-4;
paths in .env). epochs=1 is a quick run on a CPU; the paper used 30 (the default)."""
from cmb import env, preprocess, fine_tune

E = env()
preprocess(E)
model = fine_tune(E, epochs=1)
print("fine-tuned model ->", model)

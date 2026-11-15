"""Train stage 1 (U-Net) and stage 2 (classifier on out-of-fold candidates) on a folder of scans and masks
(paths in .env). epochs=1 is a quick run on a CPU; the paper used 60."""

from cmb import env, preprocess, train

E = env()
preprocess(E)
model = train(E, epochs=1)
print("model ->", model)

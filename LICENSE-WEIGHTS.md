# Licence of trained model weights

The code in this repository is MIT-licensed (see LICENSE). Trained model weights are NOT part of the git tree; the VALDO-trained
stage-1 weights of the paper are assets of the GitHub releases paper-v3-2026-11 and v1.1.0 (`cmb get-model valdo_t2`,
SHA-256 checked); release v1.1.0 also holds a stage-2 classifier trained on VALDO alone (`cmb get-model valdo_stage2`).
Weights trained on the VALDO Task 2 data are, as a precaution, released under **CC BY-NC-SA 4.0**, because the VALDO data
(https://zenodo.org/records/4520773) carry that licence and it is legally unsettled whether trained weights are "adapted material".
Non-commercial use only; cite Sudre et al., "Where is VALDO? VAscular Lesions Detection and segmentatiOn challenge at MICCAI 2021",
Medical Image Analysis (2024) 103029, doi:10.1016/j.media.2023.103029, and the VALDO funding acknowledgements listed in docs/licence-valdo.md.
Weights trained on in-house clinical data are never released here. This is not legal advice.

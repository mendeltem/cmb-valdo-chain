#!/bin/bash
# The code uses absolute paths (see README, section "Paths"). On a new machine, create the same tree under $HOME
# or point it to your own locations with symlinks. Run once:  bash scripts/link_paths.sh
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p ~/data/work ~/data/extern ~/data/bids ~/data/models/vortrainiert
[ -e ~/data/work/valdo-t2s ] || ln -s "$REPO" ~/data/work/valdo-t2s          # the code expects ROOT = ~/data/work/valdo-t2s
mkdir -p ~/data/work/mb-arena/{gitter,transfer,stage2,analysis,synthseg-t2star,synthseg-swi}   # private cohort work dir (never public)
mkdir -p "$REPO"/{dev,ergebnisse/turnier,ergebnisse/stage2,ergebnisse/analysis,logs}
echo "tree ready: ~/data/work/valdo-t2s -> $REPO ; put VALDO under ~/data/extern/valdo2021/Task2 and your BIDS cohorts under ~/data/bids/"

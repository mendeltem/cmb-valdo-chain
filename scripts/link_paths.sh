#!/bin/bash
# The case lists and manifests in dev/ hold paths RELATIVE to the repository root (dev/gitter_d/sub-XXX,
# data/valdo2021/Task2/sub-XXX/...). Run the commands from the repository root, and once:
#     CMB_VALDO=/path/to/valdo2021/Task2 bash scripts/link_paths.sh
# This links your VALDO copy to data/valdo2021/Task2 and creates the result folders. Own (non-public) cohorts live in
# CMB_PRIVATE (default: private/, ignored by git). All settings: python -m cmb.paths
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO"/{dev,ergebnisse/turnier,ergebnisse/stage2,ergebnisse/analysis,logs,data/valdo2021,models,private}
if [ -n "${CMB_VALDO:-}" ] && [ ! -e "$REPO/data/valdo2021/Task2" ]; then ln -s "$(readlink -f "$CMB_VALDO")" "$REPO/data/valdo2021/Task2"; fi
[ -e "$REPO/data/valdo2021/Task2" ] && echo "VALDO: $REPO/data/valdo2021/Task2 -> $(readlink -f "$REPO/data/valdo2021/Task2")" \
    || echo "VALDO not linked yet: CMB_VALDO=/path/to/Task2 bash scripts/link_paths.sh"
echo "tree ready under $REPO"

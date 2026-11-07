# Shared setup for the run scripts (sourced, not run). The repository root becomes the working directory; WORK is the
# folder for data, checkpoints and logs, which stay outside Git (default /tmp/Tdyssey; set TDYSSEY_WORK to change it).
WORK=${TDYSSEY_WORK:-/tmp/Tdyssey}
if [ -f "$WORK/env.sh" ]; then source "$WORK/env.sh"; fi    # optional machine settings (caches, MPS), kept outside Git
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export WORK REPO
cd "$REPO" || exit 1

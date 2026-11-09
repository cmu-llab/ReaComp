#!/usr/bin/env bash
# Autoresearch HPO domain — prepare the training environment for the real objective.
#
# Clones ferreirafabio/autoresearch (the fork the benchmark of arXiv:2603.24647 trains
# with) at the commit pinned in rewards/autoresearch_hpo.py, checks that train.py and
# prepare.py are the pinned files, installs the dependencies with uv and runs
# prepare.py (downloads climbmix-400b-shuffle shards and trains the tokenizer into
# ~/.cache/autoresearch). Needs git, uv and, for the last step, one CUDA GPU.
# Safe to re-run.
#
# Usage (from project root):
#   bash scripts/setup_autoresearch_hpo.sh
#   AUTORESEARCH_DIR=/scratch/$USER/autoresearch bash scripts/setup_autoresearch_hpo.sh
#
# Not needed for the synthetic tasks (data/autoresearch_hpo/synthetic_tasks.jsonl).
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# Where the checkout goes. The runners look in $AUTORESEARCH_DIR, then in this default.
AUTORESEARCH_DIR=${AUTORESEARCH_DIR:-original_baseline_repos/autoresearch}

# Extra arguments for prepare.py, e.g. "--num-shards 8". Empty = upstream defaults.
PREPARE_ARGS=${PREPARE_ARGS:-}
# ─────────────────────────────────────────────────────────────────────────────

# Repository, commit and file hashes come from the verifier: one source of truth.
read -r REPO COMMIT TRAIN_SHA PREPARE_SHA < <(python - <<'PY'
from rewards.autoresearch_hpo import UPSTREAM
print(UPSTREAM["train_repo"], UPSTREAM["train_commit"],
      UPSTREAM["train_py_sha256"], UPSTREAM["prepare_py_sha256"])
PY
)

echo "Autoresearch HPO setup"
echo "  repo     : https://github.com/${REPO} @ ${COMMIT}"
echo "  checkout : ${AUTORESEARCH_DIR}"
echo ""

if [ ! -d "${AUTORESEARCH_DIR}/.git" ]; then
    mkdir -p "$(dirname "${AUTORESEARCH_DIR}")"
    git clone "https://github.com/${REPO}.git" "${AUTORESEARCH_DIR}"
fi
if ! git -C "${AUTORESEARCH_DIR}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
    git -C "${AUTORESEARCH_DIR}" fetch origin
fi
git -C "${AUTORESEARCH_DIR}" checkout --detach "${COMMIT}"

check_sha() {  # check_sha <file> <expected sha256>
    local actual
    actual=$(sha256sum "$1" | cut -d' ' -f1)
    if [ "${actual}" != "$2" ]; then
        echo "ERROR: $1 has sha256 ${actual}, expected $2" >&2
        echo "       (local modifications? run: git -C ${AUTORESEARCH_DIR} status)" >&2
        exit 1
    fi
}
check_sha "${AUTORESEARCH_DIR}/train.py" "${TRAIN_SHA}"
check_sha "${AUTORESEARCH_DIR}/prepare.py" "${PREPARE_SHA}"
echo "train.py / prepare.py match the pinned sources."

cd "${AUTORESEARCH_DIR}"
uv sync
# shellcheck disable=SC2086  # PREPARE_ARGS is a list of words
uv run prepare.py ${PREPARE_ARGS}

echo ""
echo "GPU seen by the training environment:"
uv run python -c "import torch; p = torch.cuda.get_device_properties(0); print(f'  {p.name}, {p.total_memory / 2**30:.1f} GB')"
echo ""
echo "Done. The task files in data/autoresearch_hpo/ describe the paper's hardware (one"
echo "H200, 141 GB, training capped to ~76 GB); the runners refuse other GPUs unless the"
echo "task records are regenerated for them (data_generation/autoresearch_hpo_generator.py)."

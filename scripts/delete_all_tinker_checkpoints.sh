#!/usr/bin/env bash
set -euo pipefail

# Delete all remote Tinker checkpoints for the account in TINKER_API_KEY.
# Requires: tinker CLI on PATH, python

usage() {
  cat <<'USAGE'
Usage: scripts/delete_all_tinker_checkpoints.sh [--yes]

Options:
  --yes   Skip interactive confirmation

Notes:
  - Loads TINKER_API_KEY from .env if present in repo root.
  - Deletes ALL remote checkpoints in your account (irreversible).
USAGE
}

confirm=true
if [[ ${1-} == "--help" || ${1-} == "-h" ]]; then
  usage
  exit 0
fi
if [[ ${1-} == "--yes" || ${1-} == "-y" ]]; then
  confirm=false
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$repo_root/.env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$repo_root/.env"; set +a
fi

if [[ -z "${TINKER_API_KEY-}" ]]; then
  echo "TINKER_API_KEY is not set. Export it or add it to $repo_root/.env" >&2
  exit 1
fi

if ! command -v tinker >/dev/null 2>&1; then
  echo "tinker CLI not found on PATH." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found on PATH." >&2
  exit 1
fi

if $confirm; then
  echo "This will permanently delete ALL remote Tinker checkpoints for this account."
  read -r -p "Continue? (y/N) " ans
  if [[ ! "$ans" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
fi

tmp_json="$(mktemp)"
list_file="$(mktemp)"
trap 'rm -f "$tmp_json" "$list_file"' EXIT

tinker -f json checkpoint list --limit 0 > "$tmp_json"

TMPFILE="$tmp_json" python - <<'PY' > "$list_file"
import json, os
path = os.environ["TMPFILE"]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
for ckpt in data.get("checkpoints", []):
    p = ckpt.get("tinker_path")
    if p:
        print(p)
PY

if [[ ! -s "$list_file" ]]; then
  echo "No checkpoints found."
  exit 0
fi

# Delete in batches to avoid command-length limits
xargs -r -n 50 tinker checkpoint delete -y < "$list_file"

# Verify
if tinker -f json checkpoint list --limit 0 | python - <<'PY'
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
print(len(data.get("checkpoints", [])))
PY
then
  remaining=$(tinker -f json checkpoint list --limit 0 | python - <<'PY'
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)
print(len(data.get("checkpoints", [])))
PY
)
  if [[ "$remaining" == "0" ]]; then
    echo "All checkpoints deleted."
  else
    echo "Some checkpoints remain: $remaining"
  fi
else
  echo "Warning: failed to verify remaining checkpoints." >&2
fi

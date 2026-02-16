#!/usr/bin/env bash
set -euo pipefail

MODE="dry-run"
if [[ "${1:-}" == "--apply" ]]; then
  MODE="apply"
elif [[ "${1:-}" == "--dry-run" || -z "${1:-}" ]]; then
  MODE="dry-run"
else
  echo "Usage: $0 [--dry-run|--apply]" >&2
  exit 2
fi

mapfile -t TARGET_DIRS < <(
  {
    find . -type d -path './checkpoints/tinker_fim' 2>/dev/null
    find . -type d -regex '.*/tinker[^/]*/checkpoint_[^/]+' 2>/dev/null
  } | sed 's#^\./##' | sort -u
)

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  echo "No Tinker LoRA checkpoint directories found."
  exit 0
fi

echo "Found ${#TARGET_DIRS[@]} checkpoint director$( [[ ${#TARGET_DIRS[@]} -eq 1 ]] && echo 'y' || echo 'ies' ):"
printf ' - %s\n' "${TARGET_DIRS[@]}"

if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run only. Re-run with --apply to delete."
  exit 0
fi

for d in "${TARGET_DIRS[@]}"; do
  find "$d" -type f -delete
  find "$d" -depth -type d -empty -delete
  echo "Deleted: $d"
done

# Clean up empty parent folders created only for checkpoints.
find ./checkpoints -type d -empty -delete 2>/dev/null || true

echo "Done."

---
name: delete-tinker-lora-checkpoints
description: Delete local Tinker LoRA checkpoint artifacts safely. Use when the user asks to clean, remove, or purge Tinker training checkpoints or LoRA checkpoint folders.
---

# Delete Tinker LoRA Checkpoints

Use this skill when a user asks to remove local Tinker LoRA checkpoints.

## Workflow

1. Run a dry-run first:

```bash
bash skills/delete-tinker-lora-checkpoints/scripts/delete_tinker_lora_checkpoints.sh --dry-run
```

2. If the paths are correct, delete them:

```bash
bash skills/delete-tinker-lora-checkpoints/scripts/delete_tinker_lora_checkpoints.sh --apply
```

## Target Patterns

The script removes only directories under the current repo that match known Tinker checkpoint layouts:
- `checkpoints/tinker_fim`
- `**/tinker*/checkpoint_*`

It intentionally does not touch virtualenv packages or source files.

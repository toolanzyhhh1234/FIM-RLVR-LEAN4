Below is a **detailed engineering spec** you can hand to your coding AI agent to fix the `Ġ/Ċ` decode corruption in your **Ministral 14B + Lean4 GRPO (TRL)** pipeline. It’s written as an implementation-ready task doc with acceptance criteria, edge cases, and suggested patches.

***

# Spec: Fix Detokenization Corruption (`Ġ`, `Ċ`, mojibake) in TRL GRPO Lean4 Pipeline

## 0) Executive Summary

**Problem:** During GRPO training, the strings passed to the Lean reward function contain tokenization artifacts such as `Ġ` (space marker) and `Ċ` (newline marker), and sometimes mojibake sequences (e.g., `âĦķ` instead of `ℕ`). These artifacts corrupt the assembled Lean code fed into the verifier, causing verification failures.

**Root cause hypothesis:** The reward function is receiving **improperly detokenized text** (e.g., token strings or partially decoded output) from TRL’s generation→reward bridge. Similar symptoms (`Ġ/Ċ`) appear when code uses token-string joins (`convert_ids_to_tokens`) instead of decoding from token IDs; robust practice is to decode from the **token IDs** with `tokenizer.batch_decode`/`decode`. This symptom is also observed with Ministral-3 wrappers that decode with the wrong path.  
Sources: TRL GRPO trainer & reward format expectations and pipeline separation; decoding artifacts references and recommended fix (`batch_decode`); Mistral tokenizers/tekken notes. [\[github.com\]](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py), [\[stackoverflow.com\]](https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model), [\[github.com\]](https://github.com/modelscope/ms-swift/issues/7185), [\[mistralai.github.io\]](https://mistralai.github.io/mistral-common/usage/tokenizers/)

**Primary fix:** In the reward function, **ignore** the provided `completions` string if `completion_ids` are available; decode text from `completion_ids` using the same tokenizer used by the model (`tokenizer.batch_decode(..., clean_up_tokenization_spaces=False)`).

**Secondary fix (optional):** Make the dataset “conversational” (messages format) to follow the path used by the official Sudoku example, which often avoids this issue.

***

## 1) Goals / Non-Goals

### Goals

1.  Ensure reward function receives **clean, fully decoded text** (spaces/newlines/unicode preserved).
2.  Ensure assembled `verifier_input` contains **actual `\n` and spaces**, not `Ċ/Ġ`.
3.  Keep solution robust across TRL versions where kwargs naming may differ (e.g., `completion_ids`, `completions_ids`).
4.  Keep pipeline compatible with current dataset and logging setup.

### Non-Goals

*   Do not redesign the entire training pipeline or curriculum logic.
*   Do not change the underlying model/tokenizer pairing.
*   Do not implement deep Lean formatting logic beyond minimum correctness required for verifier input assembly.

***

## 2) Background / Observations

*   TRL’s GRPO supports datasets in **standard** (plain text prompt) or **conversational** (structured messages) formats. [\[github.com\]](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
*   TRL reward functions can be called with structured “completions” (chat message dicts) in some paths and/or additional keyword args. [\[huggingface.co\]](https://huggingface.co/docs/trl/main/en/rewards), [\[github.com\]](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
*   `Ġ/Ċ` artifacts commonly indicate token-string leakage; community guidance is to decode from IDs using `tokenizer.batch_decode` rather than assembling token strings manually. [\[stackoverflow.com\]](https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model)
*   Similar `Ġ/Ċ` corruption was reported specifically with **Ministral-3-14B** in other tooling wrappers, reinforcing that wrappers/bridges can mishandle decoding. [\[github.com\]](https://github.com/modelscope/ms-swift/issues/7185)
*   Mistral documents their newer tokenizers (tekken) as tiktoken-based, which increases the likelihood that “wrong decode path” yields garbage. [\[mistralai.github.io\]](https://mistralai.github.io/mistral-common/usage/tokenizers/)

***

## 3) Proposed Fix: Decode from `completion_ids` inside reward function

### 3.1 Design

Implement a **single canonical “get\_completion\_texts”** utility used by the reward function that:

1.  **Preferred path:** If token IDs exist in kwargs (any of several key names), decode them via:
    *   `tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)`
2.  **Fallback path A:** If completions are chat format (`[[{"content": "..."}]]`), extract `content`.
3.  **Fallback path B:** If completions are already list\[str], use them as-is.
4.  Add a **sanitization check** that warns if `Ġ` or `Ċ` still appear after decoding.

This fix is aligned with TRL’s reward function being a callable that receives completions plus kwargs, and with best practices for decoding. [\[huggingface.co\]](https://huggingface.co/docs/trl/main/en/rewards), [\[stackoverflow.com\]](https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model)

***

## 4) Implementation Tasks (Step-by-Step)

### Task A — Patch the Lean reward factory to decode from IDs

**File:** `train_gspo_fim_mistral3 (1).py`  
**Function:** `lean_validity_reward_factory(...)`

#### A1) Change factory signature

Current:

```python
def lean_validity_reward_factory(verifier, curriculum):
```

Change to:

```python
def lean_validity_reward_factory(verifier, curriculum, tokenizer):
```

#### A2) Add helper to extract & decode completions

Add inside the factory (or module-level helper):

```python
def _decode_completions(completions, tokenizer, **kwargs):
    # Candidate keys seen across TRL versions
    id_keys = ["completion_ids", "completions_ids", "completion_token_ids", "output_ids"]

    completion_ids = None
    for k in id_keys:
        if k in kwargs and kwargs[k] is not None:
            completion_ids = kwargs[k]
            break

    if completion_ids is not None:
        # handle torch tensors
        if hasattr(completion_ids, "tolist"):
            completion_ids = completion_ids.tolist()

        texts = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts

    # Chat-format fallback: completions = [[{"content": "..."}], ...]
    if completions and isinstance(completions[0], list) and completions[0] and isinstance(completions[0][0], dict):
        return [c[0].get("content", "") for c in completions]

    # Plain list[str] fallback
    return completions
```

#### A3) Modify reward function to use decoded texts

In `lean_validity_reward(...)` replace:

*   `generated_text` usage with `decoded_texts[idx]`
*   raw logging should log both `raw` and `decoded` if they differ

Example patch (conceptual):

```python
def lean_validity_reward(completions, fim_prefix, fim_suffix, theorem_id, task_type=None, **kwargs):
    decoded_texts = _decode_completions(completions, tokenizer, **kwargs)

    # Use decoded_texts for extraction & verifier input
    for idx, (text, prefix, suffix) in enumerate(zip(decoded_texts, fim_prefix, fim_suffix)):
        ...
```

#### A4) Add corruption detector & warning

After decoding:

```python
def _has_bpe_artifacts(s):
    return ("Ġ" in s) or ("Ċ" in s)

if any(_has_bpe_artifacts(t) for t in decoded_texts[:3]):
    print("[WARN] Decoded completions still contain Ġ/Ċ artifacts; investigate tokenizer/TRL path.")
```

#### A5) Update trainer instantiation

Change:

```python
reward_funcs=[lean_validity_reward_factory(verifier, curriculum)]
```

To:

```python
reward_funcs=[lean_validity_reward_factory(verifier, curriculum, tokenizer)]
```

***

### Task B — (Optional but recommended) Switch dataset to conversational format

**Why:** Sudoku example uses message-based completions (`completion[0]["content"]`). Your current pipeline pre-renders templates into a single string and may follow a different TRL path. TRL explicitly supports conversational datasets. [\[github.com\]](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py), [\[huggingface.co\]](https://huggingface.co/docs/trl/main/en/rewards)

**Where:** `build_dynamic_transform(...)`

#### B1) Instead of returning `prompt` as a rendered string

Current:

```python
text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompts.append(text_prompt)
```

Optional change:

*   Return the structured `messages` in the dataset as `prompt` (or `messages`) depending on TRL’s expected column naming.
*   Ensure dataset format matches TRL’s conversational schema.

**Note:** Only do this if Task A alone doesn’t fully fix it, since Task A is the robust decoding fix.

***

### Task C — Lean verifier assembly correctness (separate but important)

After decoding is fixed, you still must ensure **Lean syntax** is correct. Your example shows `rfl` inserted without the base-case pattern `| zero =>`.

#### C1) Ensure the “missing block” insertion aligns to the placeholder position

If your masked region is intended to be the entire base-case branch, the model output should be inserted as:

```lean
| zero => rfl
```

**Acceptance check:** `verifier_input` should never contain lines starting with stray whitespace + `rfl` immediately after `induction a with` unless prefixed by a case marker.

This is not a TRL decode issue, but decoding fixes will reveal it clearly.

***

## 5) Acceptance Criteria

### Primary acceptance

1.  **No `Ġ` or `Ċ`** appear in:
    *   `raw_completions.log` **decoded field** (you can keep the raw field)
    *   `verifier_input` log
2.  Unicode is preserved:
    *   `ℕ` appears as `ℕ`, not mojibake like `âĦķ`
3.  Lean verifier success rate increases (non-zero successes on known easy samples like `add_comm` base case with `rfl`).

### Secondary acceptance

4.  The reward function works whether `completions` is:
    *   list\[str]
    *   chat-format list\[list\[dict]]
5.  The reward function works whether token IDs arrive as:
    *   torch tensors
    *   python lists

***

## 6) Debug / Telemetry Requirements

Add structured logging for the first N samples per run:

*   `raw_completion_preview` (first 200 chars of whatever TRL gave in `completions`)
*   `decoded_completion_preview` (first 200 chars of decoded result)
*   boolean `artifacts_in_raw` and `artifacts_in_decoded`
*   the keys available in `kwargs` for the first reward call:
    *   `print(list(kwargs.keys()))` once

This helps confirm whether TRL is providing `completion_ids` or not.

***

## 7) Risk & Mitigations

### Risk: TRL version doesn’t pass `completion_ids`

Mitigation:

*   Use the fallback paths (chat content / list\[str]).
*   If no token IDs are passed, consider enabling the optional conversational dataset path (Task B), or patching TRL-side rollout to forward IDs (more invasive).

### Risk: Over-aggressive cleanup changes whitespace

Mitigation:

*   Use `clean_up_tokenization_spaces=False` to preserve exact spaces/newlines (important for Lean). This aligns with avoiding tokenization-space normalization. [\[stackoverflow.com\]](https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model)

### Risk: Special tokens leak into Lean code

Mitigation:

*   Always decode with `skip_special_tokens=True`
*   Optionally strip known chat template residues if present (only after decode).

***

## 8) “Definition of Done” Checklist (for your coding agent)

*   [ ] Reward factory accepts `tokenizer` and uses `_decode_completions(...)`
*   [ ] Trainer instantiation updated to pass tokenizer into factory
*   [ ] Logs show decoded completions contain real `\n` and spaces
*   [ ] `verifier_input` no longer contains `Ġ/Ċ`
*   [ ] Add-case formatting checked (e.g., `| zero => rfl`)
*   [ ] One minimal test run validates at least one proof succeeds

***

## 9) Optional: Minimal Repro Test (unit-ish)

Create a small debug function (not necessarily committed) that:

1.  Runs one `model.generate` call on a known prompt
2.  Prints:
    *   `completion_ids` (first 50 ids)
    *   `tokenizer.batch_decode(completion_ids)`
    *   the string seen by reward function

Goal: verify the pipeline mismatch is purely in the “bridge”.

***

If you want, paste your current `print(kwargs.keys())` output from inside the reward function once — I can refine the spec to **exactly** the key name TRL uses in your environment (`completion_ids` vs `completions_ids` etc.) and reduce guesswork for your coding agent.

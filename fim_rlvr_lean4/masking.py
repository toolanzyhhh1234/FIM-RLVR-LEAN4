import random


def reconstruct_full_code(raw_prompt):
    """
    Reconstruct the full code from the <PFX>...<SFX>...<MID> format.
    Returns full_code (str) or None if parsing fails.
    """
    try:
        pfx_idx = raw_prompt.find("<PFX>")
        sfx_idx = raw_prompt.find("<SFX>")
        mid_idx = raw_prompt.find("<MID>")

        if pfx_idx != -1 and sfx_idx != -1 and mid_idx != -1:
            prefix = raw_prompt[pfx_idx + 5 : sfx_idx]
            suffix = raw_prompt[sfx_idx + 5 : mid_idx]
            # The middle logic in the original script assumed the completion field
            # held the middle, so this function mainly extracts prefix and suffix.
            # We return them to be combined.
            return prefix, suffix
    except:
        return None, None
    return None, None


def apply_dynamic_mask(full_code, ratio):
    """
    Apply FIM masking to full_code with the given ratio.
    Returns (prefix, suffix, middle_truth).
    """
    lines = full_code.splitlines(keepends=True)
    total_lines = len(lines)

    # Identify the proof block (simple heuristic: after := by)
    # Ideally we'd parse, but for now we mask lines in the body.
    # Finding " := by"
    proof_start_idx = 0
    found_proof = False
    for i, line in enumerate(lines):
        if ":= by" in line:
            proof_start_idx = i + 1
            found_proof = True
            break

    if not found_proof:
        # Fallback for when ":= by" isn't found (e.g. non-tactic proof or malformed)
        # We might just consider the whole thing or fail.
        # Let's assume the whole file is the "body" if no marker,
        # BUT for Lean proofs we really want the tactic block.
        # If we can't find it, masking doesn't make much sense.
        # Let's return the whole thing as prefix (no hole) or return empty hole.
        # Returning NO hole means the model has nothing to do?
        # Let's default to masking the END if we can't find structure.
        proof_start_idx = 0

    proof_lines = lines[proof_start_idx:]
    if not proof_lines:
        return "".join(lines), "", ""

    num_proof_lines = len(proof_lines)

    if ratio >= 1.0:
        # Mask EVERYTHING in the proof.
        # "at the end the model get the question only... do not even provide the number of lines"
        prefix = "".join(lines[:proof_start_idx])
        suffix = ""  # No suffix provided
        middle = "".join(proof_lines)
        return prefix, suffix, middle

    # Partial masking
    num_to_mask = max(1, int(num_proof_lines * ratio))

    # We want to mask a contiguous block or random?
    # Standard FIM often does random span. The spec says "sample a hole location".
    # Let's pick a random start.
    max_start = max(0, num_proof_lines - num_to_mask)
    start_offset = random.randint(0, max_start)

    masked_slice = proof_lines[start_offset : start_offset + num_to_mask]

    prefix = "".join(lines[: proof_start_idx + start_offset])
    # suffix is the rest
    suffix = "".join(proof_lines[start_offset + num_to_mask :])
    middle = "".join(masked_slice)

    return prefix, suffix, middle

import random
from typing import List, Tuple, Optional


def split_keepends(s: str) -> List[str]:
    return s.splitlines(keepends=True)


def generate_exact_line_fim(text: str, ratio: float = 0.15) -> Tuple[str, str, str]:
    lines = split_keepends(text)
    if len(lines) < 3:
        return "", "", ""

    k = max(1, int(len(lines) * ratio))
    k = min(k, len(lines))

    if len(lines) - k >= 0:
        start_idx = random.randint(0, len(lines) - k)
    else:
        return "", "", ""

    end_idx = start_idx + k

    prefix = "".join(lines[:start_idx])
    middle = "".join(lines[start_idx:end_idx])
    suffix = "".join(lines[end_idx:])

    return prefix, middle, suffix


def split_proof_body(code: str) -> Optional[Tuple[str, str]]:
    splitter = ":= by"
    idx = code.find(splitter)
    if idx == -1:
        return None

    cut = idx + len(splitter)
    header = code[:cut]
    proof_body = code[cut:]
    return header, proof_body


def generate_proof_only_exact_fim(code: str, ratio: float = 0.15) -> Tuple[str, str, str]:
    split = split_proof_body(code)
    if split is None:
        return "", "", ""

    header, proof_body = split
    prefix_body, middle, suffix_body = generate_exact_line_fim(proof_body, ratio=ratio)
    if not middle:
        return "", "", ""

    prefix = header + prefix_body
    suffix = suffix_body
    return prefix, middle, suffix

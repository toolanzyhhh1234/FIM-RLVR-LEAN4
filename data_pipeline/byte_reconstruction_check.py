import argparse
import json
from typing import Optional, Tuple

from datasets import load_dataset

from data_pipeline import fim_generator
from data_pipeline import fim_generator_fixed
from data_pipeline import fim_simple_fix


def first_diff(a: str, b: str) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def diff_context(a: str, b: str, idx: int, window: int = 60) -> Tuple[str, str]:
    start = max(0, idx - window)
    end = min(max(len(a), len(b)), idx + window)
    return a[start:end], b[start:end]


def concat_with_newline(a: str, b: str) -> str:
    if not a:
        return b
    if not b:
        return a
    if a.endswith("\n") or b.startswith("\n"):
        return a + b
    return a + "\n" + b


def build_fim_pieces(code: str, mode: str, ratio: float) -> Optional[Tuple[str, str, str]]:
    if mode == "proof_only_original":
        splitter = ":= by"
        if splitter not in code:
            return None
        parts = code.split(splitter, 1)
        header = parts[0] + splitter
        proof_body = parts[1]
        prefix_body, middle, suffix_body = fim_generator.generate_fim_sample(proof_body, ratio=ratio)
        if not middle:
            return None
        prefix = concat_with_newline(header, prefix_body)
        suffix = suffix_body
        return prefix, middle, suffix

    if mode == "proof_only_fixed":
        splitter = ":= by"
        if splitter not in code:
            return None
        parts = code.split(splitter, 1)
        header = parts[0] + splitter
        proof_body = parts[1]
        prefix_body, middle, suffix_body = fim_generator_fixed.generate_fim_sample(proof_body, ratio=ratio)
        if not middle:
            return None
        prefix = concat_with_newline(header, prefix_body)
        suffix = suffix_body
        return prefix, middle, suffix

    if mode == "file_simple":
        prefix, middle, suffix = fim_simple_fix.generate_simple_fim(code, ratio=ratio)
        if not middle:
            return None
        return prefix, middle, suffix

    raise ValueError(f"Unknown mode: {mode}")


def has_glued_boundary(prefix: str, middle: str, suffix: str) -> bool:
    if prefix and middle:
        if not prefix.endswith("\n") and not middle.startswith("\n"):
            return True
    if middle and suffix:
        if not middle.endswith("\n") and not suffix.startswith("\n"):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["proof_only_original", "proof_only_fixed", "file_simple"], default="proof_only_fixed")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--min_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump_mismatches", type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset("AI-MO/NuminaMath-LEAN", split="train", streaming=True)

    exact = 0
    normalized = 0
    glue = 0
    tested = 0
    dumped = 0

    for ex in ds.shuffle(seed=args.seed):
        if tested >= args.n:
            break

        if ex.get("ground_truth_type") != "complete":
            continue

        code = ex.get("formal_ground_truth") or ""
        if len(code.strip()) < args.min_len:
            continue

        pieces = build_fim_pieces(code, mode=args.mode, ratio=args.ratio)
        if pieces is None:
            continue

        prefix, middle, suffix = pieces
        reconstructed = prefix + middle + suffix

        if has_glued_boundary(prefix, middle, suffix):
            glue += 1

        tested += 1

        if reconstructed == code:
            exact += 1
            normalized += 1
            continue

        if reconstructed.rstrip("\n") == code.rstrip("\n"):
            normalized += 1

        if dumped < args.dump_mismatches:
            idx = first_diff(code, reconstructed)
            payload = {
                "mode": args.mode,
                "len_original": len(code),
                "len_reconstructed": len(reconstructed),
                "first_diff": idx,
            }
            if idx is not None:
                a_ctx, b_ctx = diff_context(code, reconstructed, idx)
                payload["original_ctx"] = a_ctx
                payload["reconstructed_ctx"] = b_ctx
            print(json.dumps(payload, ensure_ascii=False))
            dumped += 1

    print(
        json.dumps(
            {
                "mode": args.mode,
                "tested": tested,
                "exact_match": exact,
                "exact_match_rate": (exact / tested) if tested else 0.0,
                "normalized_match": normalized,
                "normalized_match_rate": (normalized / tested) if tested else 0.0,
                "glued_boundary": glue,
                "glued_boundary_rate": (glue / tested) if tested else 0.0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

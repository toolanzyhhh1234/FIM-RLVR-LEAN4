#!/usr/bin/env python3
"""Test script to diagnose tokenizer decoding issues with Ministral 3."""

from unsloth import FastVisionModel

print("Loading model...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Ministral-3-3B-Instruct-2512",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("\n" + "=" * 50)
print("TOKENIZER DECODING TEST")
print("=" * 50)

# Handle PixtralProcessor vs regular tokenizer
print(f"\nProcessor/Tokenizer class: {type(tokenizer).__name__}")

# Get the actual tokenizer from the processor if needed
if hasattr(tokenizer, 'tokenizer'):
    actual_tokenizer = tokenizer.tokenizer
    print(f"Inner tokenizer class: {type(actual_tokenizer).__name__}")
else:
    actual_tokenizer = tokenizer

# Test encoding and decoding
test_text = "  induction _; goal\n"
tokens = actual_tokenizer.encode(test_text)
decoded = actual_tokenizer.decode(tokens)
decoded_clean = actual_tokenizer.decode(tokens, skip_special_tokens=True)

print(f"\nOriginal text: {repr(test_text)}")
print(f"Token IDs: {tokens}")
print(f"Decoded: {repr(decoded)}")
print(f"Decoded (skip_special=True): {repr(decoded_clean)}")

# Check tokenizer type
print(f"\nTokenizer vocab size: {actual_tokenizer.vocab_size}")

# Check for byte-level BPE artifacts
if "Ġ" in decoded or "Ċ" in decoded:
    print("\n⚠️  WARNING: Byte-level BPE artifacts detected in decoded output!")
    print("   This indicates the tokenizer is not properly decoding byte tokens.")
else:
    print("\n✓ Tokenizer decoding looks correct.")

# Test with the problematic characters from logs
print("\n" + "=" * 50)
print("TESTING PROBLEMATIC PATTERNS FROM LOGS")
print("=" * 50)

# This is what we see in the logs - let's see if we can decode it
problematic = "ĊĠĠinductionĠ_;ĠgoalĊ"
print(f"\nProblematic string from logs: {repr(problematic)}")

# Try to understand the byte mapping
print("\nByte analysis of problematic characters:")
print(f"  'Ċ' (U+010A) = {ord('Ċ')} = should be newline (10)")
print(f"  'Ġ' (U+0120) = {ord('Ġ')} = should be space (32)")

# Manual decode attempt
manual_decoded = problematic.replace("Ċ", "\n").replace("Ġ", " ")
print(f"\nManual decode: {repr(manual_decoded)}")

# Test with the natural number symbol that's causing issues
print("\n" + "=" * 50)
print("TESTING NATURAL NUMBER SYMBOL (ℕ)")
print("=" * 50)

test_nat = "theorem add_comm (a b : ℕ) : a + b = b + a"
print(f"\nOriginal: {repr(test_nat)}")
print(f"ℕ character: U+{ord('ℕ'):04X} = {ord('ℕ')}")
print(f"UTF-8 bytes for ℕ: {[hex(b) for b in 'ℕ'.encode('utf-8')]}")

tokens_nat = actual_tokenizer.encode(test_nat)
decoded_nat = actual_tokenizer.decode(tokens_nat)
print(f"\nToken IDs: {tokens_nat}")
print(f"Decoded: {repr(decoded_nat)}")

# Show any problematic characters
print("\nNon-ASCII character analysis of decoded output:")
for i, char in enumerate(decoded_nat):
    if ord(char) > 127:
        print(f"  Position {i}: {repr(char)} = U+{ord(char):04X} ({ord(char)})")

# Test the byte_to_unicode mapping used by GPT-2 style tokenizers
print("\n" + "=" * 50)
print("BYTE TO UNICODE MAPPING TEST")
print("=" * 50)

def bytes_to_unicode():
    """GPT-2 style byte to unicode mapping."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# Show what ℕ's UTF-8 bytes would encode to
nat_bytes = 'ℕ'.encode('utf-8')
print(f"\nℕ UTF-8 bytes: {list(nat_bytes)}")
print("Expected BPE encoding for each byte:")
for b in nat_bytes:
    if b in byte_encoder:
        print(f"  Byte {b} (0x{b:02X}) -> {repr(byte_encoder[b])} (U+{ord(byte_encoder[b]):04X})")

# Try to decode the problematic â&7 pattern
print("\n" + "=" * 50)
print("ANALYZING 'â&7' PATTERN")
print("=" * 50)
problematic_nat = "â&7"
print(f"Problematic string: {repr(problematic_nat)}")
print("Character codes:", [f"U+{ord(c):04X}" for c in problematic_nat])

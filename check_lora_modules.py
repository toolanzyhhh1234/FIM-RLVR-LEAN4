from unsloth import FastLanguageModel
import torch

max_seq_length = 768
lora_rank = 4

print("Loading model...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        offload_embedding=True,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("\n--- CHECK 1: Count LoRA modules actually created ---")
lora_module_names = [n for n, m in model.named_modules() if "lora_" in n.lower()]
print("LoRA submodules:", len(lora_module_names))
print("\n".join(lora_module_names[:50]))

print("\n--- CHECK 2: Check trainable parameter count ---")
trainable = 0
total = 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable: {trainable:,} / {total:,} ({trainable / total:.6%})")

print("\n--- CHECK 3: Inspect actual projection names in the model ---")
cands = []
for name, mod in model.named_modules():
    # checking for common linear layer names
    if any(
        k in name
        for k in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "qkv",
            "out",
            "gate",
            "up",
            "down",
            "mlp",
        ]
    ):
        if hasattr(mod, "weight"):
            cands.append(name)
print("\n".join(cands[:200]))
print("Total candidates:", len(cands))

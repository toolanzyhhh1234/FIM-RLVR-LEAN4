import glob
import os

from transformers import AutoConfig, AutoModelForCausalLM


def main():
    # Force offline + CPU-only to avoid GPU init overhead.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    base = (
        "/root/.cache/huggingface/hub/"
        "models--unsloth--qwen3-30b-a3b-thinking-2507/snapshots"
    )
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    if not snaps:
        raise SystemExit(f"No snapshots found in {base}")

    model_dir = snaps[-1]
    print(f"Using snapshot: {model_dir}")

    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    # Build model without allocating weights if possible.
    try:
        from transformers.modeling_utils import init_empty_weights
    except Exception as exc:
        print(f"init_empty_weights not available: {exc}")
        print("Falling back to real init; may be slower.")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    else:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    moe_layer = None
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if ("Moe" in cls or "MoE" in cls) and hasattr(module, "mlp"):
            moe_layer = (name, module)
            break

    if moe_layer is None:
        raise SystemExit("No MoE module with .mlp found.")

    name, module = moe_layer
    print(f"Found MoE module: {name} ({module.__class__.__name__})")
    mlp = module.mlp
    print(f"MLP class: {mlp.__class__.__name__}")

    attrs = [a for a in dir(mlp) if not a.startswith("_")]
    key_attrs = [
        a for a in attrs if any(k in a.lower() for k in ["proj", "gate", "up", "down", "expert", "router"])
    ]
    print("MLP key attributes:", key_attrs)

    m_attrs = [a for a in dir(module) if not a.startswith("_")]
    key_m_attrs = [
        a
        for a in m_attrs
        if any(k in a.lower() for k in ["proj", "gate", "up", "down", "expert", "router", "mlp"])
    ]
    print("MoE module key attributes:", key_m_attrs)


if __name__ == "__main__":
    main()

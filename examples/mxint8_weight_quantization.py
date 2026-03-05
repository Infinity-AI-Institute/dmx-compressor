"""
MXINT8 weight quantization example using Qwen3-0.6B.

Directly quantizes Linear layer weights to MXINT8 (OCP Microscaling) format
using dmx's MXINT.cast(), then saves the quantized model. This bypasses the
dmx FX tracing pipeline, which doesn't support all model architectures.

Usage:
    python examples/mxint8_weight_quantization.py

What this does:
    - MXINT8 block quantization: shared exponent per 32-element block, 8-bit mantissa
    - Weights are cast to MXINT8-representable values (stored as FP32)
    - Identical math to what dmx's fold_weights_and_biases() produces
    - No calibration needed — MXINT8 uses per-block dynamic scaling
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dmx.compressor.numerical.format import MXINT

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The future of AI hardware is"
OUTPUT_DIR = "./qwen3-0.6b-mxint8"

# --- Step 1: Load model ---
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()


def generate(model, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# --- Step 2: FP32 baseline ---
baseline_text = generate(model, PROMPT)
print(f"\n{'FP32 baseline':>20}: {baseline_text}")

# --- Step 3: Quantize all Linear weights to MXINT8 ---
mxint8 = MXINT(precision=8, block_size=32)
num_quantized = 0

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.weight.data = mxint8.cast(module.weight.data, block_dim=-1)
        num_quantized += 1

print(f"\nQuantized {num_quantized} Linear layers to MXINT8 (block_size=32)")

# --- Step 4: Generate with quantized weights ---
quantized_text = generate(model, PROMPT)
print(f"{'MXINT8 quantized':>20}: {quantized_text}")

# --- Step 5: Save quantized model ---
print(f"\nSaving quantized model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Step 6: Reload and verify ---
print("Reloading saved model to verify...")
reloaded_model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, dtype=torch.float32)
reloaded_model.eval()

reloaded_text = generate(reloaded_model, PROMPT)
print(f"{'Reloaded model':>20}: {reloaded_text}")

assert reloaded_text == quantized_text, "MISMATCH: reloaded model output differs!"
print("\nVerification passed: reloaded model produces identical output.")

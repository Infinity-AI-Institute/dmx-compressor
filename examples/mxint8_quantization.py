"""
MXINT8 quantization example using OPT-125m.

Demonstrates how to quantize a HuggingFace model to MXINT8 (OCP Microscaling)
format at block sizes 128, 64, and 32 — the formats natively supported by
d-Matrix Corsair hardware.

Usage:
    python examples/mxint8_quantization.py

Key takeaways:
    - MXINT8 should target Linear and ActActMatMul only (not LayerNorm/Softmax)
    - No calibration needed — MXINT8 uses per-block dynamic scaling
    - Smaller block sizes give better precision at the cost of more scale overhead
"""

import torch
from dmx.compressor.modeling.hf import pipeline
from dmx.compressor.modeling import DmxConfigRule, nn
from dmx.compressor import config_rules

MODEL_ID = "facebook/opt-125m"
PROMPT = "The future of AI hardware is"


def generate(pipe, prompt, max_new_tokens=50):
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]


def make_mxint8_rules(block_size):
    """Create MXINT8 quantization rules for Linear and MatMul layers.

    Only targets compute-heavy ops — normalization and softmax layers
    are left in full precision to preserve output quality.
    """
    fmt = f"MXINT8{{{block_size}}}"
    return (
        DmxConfigRule(
            module_types=(nn.Linear,),
            module_config=dict(
                input_formats=[fmt],
                weight_format=fmt,
                output_formats=[fmt],
            ),
        ),
        DmxConfigRule(
            module_types=(nn.ActActMatMul,),
            module_config=dict(
                input_formats=[fmt, fmt],
                output_formats=[fmt],
            ),
        ),
    )


# --- Step 1: Load model with baseline (no quantization) ---
print(f"Loading {MODEL_ID}...")
pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    dmx_config="BASELINE",
    device_map="cpu",
)

# --- Step 2: Generate FP32 baseline ---
baseline = generate(pipe, PROMPT)
print(f"{'FP32 baseline':>20}: {baseline}\n")

# --- Step 3: MXINT8 at each block size ---
for block_size in [128, 64, 32]:
    # Reset to baseline before each configuration
    pipe.model.configure(None, *config_rules.BASELINE)
    pipe.model.configure(None, *make_mxint8_rules(block_size))

    text = generate(pipe, PROMPT)
    print(f"{'MXINT8 k=' + str(block_size):>20}: {text}")

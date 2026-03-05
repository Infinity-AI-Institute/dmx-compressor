"""
MXINT8 weight quantization example using Qwen3-0.6B.

Quantizes Linear layer weights to packed MXINT8 format (int8 mantissas +
uint8 shared exponents per block), saves as safetensors, and verifies
lossless roundtrip. Non-Linear parameters (embeddings, norms) are kept
in FP16.

Usage:
    python examples/mxint8_weight_quantization.py

Output format (safetensors):
    {layer}.mantissas  - int8, shape [Cout, Cin]
    {layer}.scales     - uint8 shared exponent per block, shape [Cout, Cin // block_size]
    {layer}.bias       - float16 (if present)
    other params       - float16 (embeddings, norms, etc.)
"""

import json
import os

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from dmx.compressor.numerical.format import MXINT

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The future of AI hardware is"
OUTPUT_DIR = "./qwen3-0.6b-mxint8"
BLOCK_SIZE = 32
PRECISION = 8  # total bits per element (1 sign + 7 mantissa)


def generate(model, tokenizer, prompt, max_new_tokens=50):
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


# ── MXINT8 packing ──────────────────────────────────────────────────────────


def pack_mxint8(weight, block_size=BLOCK_SIZE):
    """Pack a FP32 weight tensor into MXINT8: (uint8 scales, int8 mantissas).

    For each block of `block_size` elements along dim=-1:
      - shared_exp = floor(log2(max(|x|))) in the block
      - scale = 2^(shared_exp - 6)   (6 = precision - 2)
      - mantissa_i = clamp(round(x_i / scale), -127, 127)

    Returns:
        scales:    uint8 [Cout, Cin // block_size] — biased shared exponent (bias=127)
        mantissas: int8  [Cout, Cin]
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    Cout, Cin = weight.shape
    assert Cin % block_size == 0, f"Cin={Cin} not divisible by block_size={block_size}"

    blocks = weight.float().reshape(Cout, Cin // block_size, block_size)
    max_abs = blocks.abs().amax(dim=-1)  # [Cout, num_blocks]

    # Shared exponent per block (unbiased); zero blocks get exp=0
    shared_exp = torch.where(
        max_abs > 0,
        torch.floor(torch.log2(max_abs)),
        torch.zeros_like(max_abs),
    )

    # Quantize mantissas: scale = 2^(shared_exp - 6)
    scale = (2.0 ** (shared_exp - 6)).unsqueeze(-1)  # [Cout, num_blocks, 1]
    mantissas = torch.clamp(torch.round(blocks / scale), -127, 127).to(torch.int8)

    # Bias the exponent for uint8 storage (OCP MX uses E8M0 with bias=127)
    scales = (shared_exp + 127).to(torch.uint8)

    return scales, mantissas.reshape(Cout, Cin)


def unpack_mxint8(scales, mantissas, block_size=BLOCK_SIZE):
    """Unpack MXINT8 (uint8 scales, int8 mantissas) back to FP32."""
    Cout, Cin = mantissas.shape
    shared_exp = scales.float() - 127  # remove bias
    scale = (2.0 ** (shared_exp - 6)).unsqueeze(-1)
    blocks = mantissas.reshape(Cout, Cin // block_size, block_size).float() * scale
    return blocks.reshape(Cout, Cin)


# ── Main ─────────────────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

# --- FP32 baseline ---
baseline_text = generate(model, tokenizer, PROMPT)
print(f"\n{'FP32 baseline':>20}: {baseline_text}")

# --- Quantize & pack all Linear weights ---
mxint8 = MXINT(precision=PRECISION, block_size=BLOCK_SIZE)
packed_tensors = {}
num_quantized = 0
num_roundtrip_mismatches = 0

for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue

    w = module.weight.data
    # Use MXINT.cast() as ground truth for the quantized values
    w_quantized = mxint8.cast(w, block_dim=-1)

    # Pack into int8 mantissas + uint8 scales
    scales, mantissas = pack_mxint8(w)

    # Verify roundtrip: unpack must match MXINT.cast() exactly
    w_roundtrip = unpack_mxint8(scales, mantissas)
    if not torch.equal(w_quantized, w_roundtrip):
        max_err = (w_quantized - w_roundtrip).abs().max().item()
        num_roundtrip_mismatches += 1
        print(f"  WARNING: {name} roundtrip mismatch, max error={max_err:.2e}")

    # Store packed representation
    packed_tensors[f"{name}.mantissas"] = mantissas
    packed_tensors[f"{name}.scales"] = scales
    if module.bias is not None:
        packed_tensors[f"{name}.bias"] = module.bias.data.half()

    # Apply quantized weights to model for generation test
    module.weight.data = w_quantized
    num_quantized += 1

print(f"\nPacked {num_quantized} Linear layers to MXINT8 (block_size={BLOCK_SIZE})")
if num_roundtrip_mismatches == 0:
    print("All layers pass lossless roundtrip check (pack → unpack == MXINT.cast)")
else:
    print(f"WARNING: {num_roundtrip_mismatches} layers had roundtrip mismatches")

# --- Generate with quantized weights ---
quantized_text = generate(model, tokenizer, PROMPT)
print(f"\n{'MXINT8 quantized':>20}: {quantized_text}")

# --- Save packed model ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Non-Linear parameters in FP16
for name, param in model.named_parameters():
    key_prefix = name.replace(".weight", "").replace(".bias", "")
    # Skip params already packed as Linear weights/biases
    if f"{key_prefix}.mantissas" in packed_tensors or f"{key_prefix}.bias" in packed_tensors:
        continue
    packed_tensors[name] = param.data.half()

print(f"\nSaving packed MXINT8 model to {OUTPUT_DIR}...")
save_file(packed_tensors, os.path.join(OUTPUT_DIR, "model.safetensors"))
tokenizer.save_pretrained(OUTPUT_DIR)
model.config.save_pretrained(OUTPUT_DIR)

# Save quantization metadata
quant_config = {
    "quant_method": "mxint8",
    "precision": PRECISION,
    "block_size": BLOCK_SIZE,
    "scale_format": "uint8_e8m0_bias127",
    "mantissa_format": "int8_symmetric",
    "quantized_layers": "torch.nn.Linear",
    "non_quantized_dtype": "float16",
}
with open(os.path.join(OUTPUT_DIR, "quant_config.json"), "w") as f:
    json.dump(quant_config, f, indent=2)

# --- Verify reload ---
print("Reloading and verifying...")
loaded = load_file(os.path.join(OUTPUT_DIR, "model.safetensors"))

# Spot-check: reconstruct first Linear layer and compare
first_linear = next(n for n, _ in model.named_modules() if isinstance(_, torch.nn.Linear))
w_reloaded = unpack_mxint8(loaded[f"{first_linear}.scales"], loaded[f"{first_linear}.mantissas"])
w_model = dict(model.named_modules())[first_linear].weight.data

assert torch.equal(w_reloaded, w_model), "MISMATCH: reloaded weights differ!"
print(f"Verified: {first_linear} weights match after reload.")

# Report size
fp32_size = sum(p.numel() * 4 for p in model.parameters()) / 1e6
packed_size = os.path.getsize(os.path.join(OUTPUT_DIR, "model.safetensors")) / 1e6
print(f"\nFP32 model size:    {fp32_size:.1f} MB (in memory)")
print(f"Packed MXINT8 size: {packed_size:.1f} MB (on disk)")
print(f"Compression ratio:  {fp32_size / packed_size:.2f}x")

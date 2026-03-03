import pytest
import torch
from dmx.compressor import numerical
from dmx.compressor.modeling import nn as dmxnn


RANDOM_SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)


@pytest.mark.parametrize("block_size", (32, 64, 128))
def test_castto_mxint8(block_size):
    """MXINT8 cast round-trips with bounded error."""
    x = torch.randn(1, 256, device=device)
    fmt = f"MXINT8{{{block_size}}}"
    _x = numerical.CastTo(format=fmt)(x)
    assert _x.shape == x.shape
    assert torch.isfinite(_x).all()
    # MXINT8: 7-bit mantissa + sign, shared 8-bit exponent per block
    assert torch.allclose(_x, x, rtol=0.0, atol=x.abs().max() * 2**-6)


@pytest.mark.parametrize("block_size", (32, 64, 128))
def test_mxint8_linear_forward(block_size):
    """Linear layer produces finite output under MXINT8 quantization."""
    fmt = f"MXINT8{{{block_size}}}"
    mod = dmxnn.Linear(128, 64, device=device)
    mod.configure(dict(
        input_formats=[fmt],
        weight_format=fmt,
        output_formats=[fmt],
    ))
    x = torch.randn(1, 128, device=device)
    out = mod(x)
    assert out.shape == (1, 64)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("block_size", (32, 64, 128))
def test_mxint8_actactmatmul_forward(block_size):
    """ActActMatMul produces finite output under MXINT8 quantization."""
    fmt = f"MXINT8{{{block_size}}}"
    mod = dmxnn.ActActMatMul()
    mod.configure(dict(
        input_formats=[fmt, fmt],
        output_formats=[fmt],
    ))
    a = torch.randn(1, 32, 64, device=device)
    b = torch.randn(1, 64, 32, device=device)
    out = mod(a, b)
    assert out.shape == (1, 32, 32)
    assert torch.isfinite(out).all()


def test_mxint8_preserves_zeros():
    """MXINT8 preserves exact zeros."""
    x = torch.zeros(1, 128, device=device)
    _x = numerical.CastTo(format="MXINT8{32}")(x)
    assert torch.all(_x == 0)


def test_mxint8_block_size_aliases():
    """All standard MXINT8 block size aliases are valid formats."""
    for block_size in (32, 64, 128):
        fmt = f"MXINT8{{{block_size}}}"
        cast = numerical.CastTo(format=fmt)
        x = torch.randn(1, 128, device=device)
        out = cast(x)
        assert out.shape == x.shape

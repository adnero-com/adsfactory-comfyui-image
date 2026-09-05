# LOCAL SHIM (AdsFactory) — see __init__.py in this directory. usp_attn_forward is
# the xfuser sequence-parallel attention patch applied only by the multi-GPU path.

def usp_attn_forward(*args, **kwargs):
    raise RuntimeError(
        "echomimic_v3 usp_attn_forward: xfuser sequence-parallel (multi-GPU) support "
        "is not installed in this image — single-GPU inference only.")

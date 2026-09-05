# LOCAL SHIM (AdsFactory) — not part of upstream smthemex/ComfyUI_EchoMimic.
#
# Upstream main (3a36b00) imports `from .dist import ...` in
# wan_transformer3d_audio_2512.py (the V3-flash transformer, imported at node-pack
# load), but does NOT ship the echomimic_v3/src/dist package — so the whole pack
# fails to register ("No module named ...echomimic_v3.src.dist"). The real module
# (from alibaba VideoX-Fun) only wraps xfuser sequence-parallel helpers used for
# MULTI-GPU inference; on this single-5090 box they are never called. This shim
# provides the imported names with lazy failures so single-GPU inference works and
# any accidental multi-GPU call fails loudly instead of silently.
#
# Delete this directory if upstream ever ships its own src/dist.

def _unavailable(name):
    def _raise(*args, **kwargs):
        raise RuntimeError(
            f"echomimic_v3 {name}: xfuser sequence-parallel (multi-GPU) support is "
            "not installed in this image — single-GPU inference only.")
    return _raise


try:  # pragma: no cover - only present if someone installs xfuser later
    from xfuser.core.distributed import (get_sequence_parallel_rank,
                                         get_sequence_parallel_world_size,
                                         get_sp_group)
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
except Exception:  # xfuser absent (the normal case here)
    get_sequence_parallel_rank = _unavailable("get_sequence_parallel_rank")
    get_sequence_parallel_world_size = _unavailable("get_sequence_parallel_world_size")
    get_sp_group = _unavailable("get_sp_group")
    xFuserLongContextAttention = _unavailable("xFuserLongContextAttention")

"""Windows-CPU hardening: torch.compile -> eager no-op when inductor is broken.

transformers' ModernBERT uses @torch.compile at class-definition time, which
imports torch._inductor -> stale triton and crashes on boxes where triton
can't install (win/py3.12). Inductor can't run on Windows-CPU anyway, so
eager is behavior-preserving for inference. Import this BEFORE laya.
"""
try:
    import torch
    try:
        import torch._inductor.runtime.hints  # noqa: F401
    except Exception:
        def _eager_compile(model=None, *args, **kwargs):
            return (lambda m: m) if model is None else model

        torch.compile = _eager_compile
except ImportError:
    pass

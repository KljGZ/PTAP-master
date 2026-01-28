"""Backdoor evaluation helpers (lightweight)."""

# Re-export common builders for convenience
try:
    from .dwt_eval import build_dwt_bd_dataset  # noqa: F401
except Exception:
    pass

"""Synthetic random disease generators and stress-testing benchmarks (lazy imports)."""
__all__ = ["DISEASE_PROFILES", "generate_single_disease_dataset", "run_multi_dataset_benchmark"]

def __getattr__(name):
    if name in __all__:
        from importlib import import_module
        return getattr(import_module(".generator", __name__), name)
    raise AttributeError(name)

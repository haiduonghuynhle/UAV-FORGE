import importlib as _il

__all__ = ['misc', 'geometry', 'image', 'io', 'metrics', 'plot', 'pose']

def __getattr__(name):
    if name in __all__:
        return _il.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

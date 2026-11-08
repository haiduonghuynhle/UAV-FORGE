"""Bundled YAML configurations for the native AnyVisLoc baseline adapter."""
import pathlib, yaml

_CONFIG_DIR = pathlib.Path(__file__).parent


def load_benchmark_config(region: str) -> dict:
    """Load and merge base + region config for native AnyVisLoc.

    Returns a single merged dict ready for use by benchmark_utils functions.
    """
    base_yaml   = _CONFIG_DIR / "config.yaml"
    region_yaml = _CONFIG_DIR / "Regions_params" / f"{region}.yaml"

    with open(base_yaml) as f:
        base = yaml.safe_load(f)
    with open(region_yaml) as f:
        region_cfg = yaml.safe_load(f)

    region_cfg.update(base)   # base values fill in what region doesn't define
    return region_cfg

# project_config.py
import yaml
import torch
from pathlib import Path
from types import SimpleNamespace


def load_config():
    """Loads YAML config and resolves relative paths from project root."""
    root = Path(__file__).resolve().parent 
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"CONFIG FILE LOADED: \n{cfg}")
    # inject absolute root_dir
    cfg.setdefault("project", {})
    cfg["project"]["root_dir"] = str(root)

    # detect device (gpu / cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg["project"]["device"] = device

    # makes all entries under 'paths' absolute
    for key, rel_path in cfg.get("paths", {}).items():
        if rel_path is not None:
            cfg["paths"][key] = str((root / rel_path).resolve())
    
    # Return config object with dot access
    return _to_ns(cfg)


# Enables dot access
def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ _to_ns(v) for v in obj ]
    return obj
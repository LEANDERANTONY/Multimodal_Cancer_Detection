
import os, yaml

def load_paths(cfg_path='configs/paths.yaml'):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        cfg[k] = os.path.abspath(v)
        os.makedirs(cfg[k], exist_ok=True)
    return cfg

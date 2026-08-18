# Derived from Earthformer (Apache License 2.0).
# Modified for HistCastNet.
import os
from omegaconf import OmegaConf

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


cfg = OmegaConf.create()
cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..", "..")))
cfg.datasets_dir = os.path.abspath(
    os.environ.get("HISTCASTNET_DATA_DIR", os.path.join(cfg.root_dir, "datasets"))
)  # Override with HISTCASTNET_DATA_DIR for datasets stored elsewhere.
cfg.pretrained_checkpoints_dir = os.path.join(
    cfg.root_dir, "pretrained_checkpoints"
)  # default directory for saving and loading pretrained checkpoints
cfg.exps_dir = os.path.join(
    cfg.root_dir, "experiments"
)  # default directory for saving experiment results
os.makedirs(cfg.exps_dir, exist_ok=True)

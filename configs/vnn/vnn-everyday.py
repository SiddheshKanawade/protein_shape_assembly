import os

from multi_part_assembly.utils import merge_cfg
from yacs.config import CfgNode as CN

_base_ = {
    "exp": "../_base_/default_exp.py",
    "data": "../_base_/datasets/breaking_bad/everyday.py",
    "optimizer": "../_base_/schedules/adam_cosine.py",
    "model": "../_base_/models/vnn.py",
    "loss": "../_base_/models/loss/vnn_loss.py",
}

# Miscellaneous configs
_C = CN()


def get_cfg_defaults():
    base_cfg = _C.clone()
    cfg = merge_cfg(base_cfg, os.path.dirname(__file__), _base_)
    return cfg

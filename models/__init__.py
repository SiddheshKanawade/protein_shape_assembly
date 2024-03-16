from .modules import *
from .vnn import VNNModel


def build_model(cfg):
    if cfg.model.name == 'vnn':
        return VNNModel(cfg)

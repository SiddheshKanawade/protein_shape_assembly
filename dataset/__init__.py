from .geometry_data import build_geometry_dataloader


def build_dataloader(cfg):
    if cfg.data.dataset == "geometry":
        return build_geometry_dataloader(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset} not supported")

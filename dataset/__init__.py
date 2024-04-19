from .geometry_data import build_geometry_dataloader
from .protein_data import build_geometry_dataloader_protein


def build_dataloader(cfg):
    if cfg.data.dataset == "geometry":
        return build_geometry_dataloader(cfg)
    elif cfg.data.dataset == "protein":
        return build_geometry_dataloader_protein(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.data.dataset} not supported")

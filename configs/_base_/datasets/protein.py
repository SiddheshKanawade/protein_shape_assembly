"""Everyday subset from the Breaking Bad dataset."""

from yacs.config import CfgNode as CN

_C = CN()
_C.dataset = "protein"
_C.data_dir = "protein_data"
_C.data_fn = "data_split/{}.txt"
_C.data_keys = ("part_ids",)
_C.category = "all"  # empty means all categories
_C.rot_range = -1.0  # rotation range for curriculum learning
_C.num_pc_points = 512  # points per part
_C.min_num_part = 2
_C.max_num_part = 45
_C.shuffle_parts = True
_C.overfit = -1
# _C.all_category = [
#     'BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate', 'Spoon',
#     'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie', 'DrinkBottle',
#     'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass'
# ]
_C.all_category = ["all"]
_C.colors = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 204, 0],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]


def get_cfg_defaults():
    return _C.clone()

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ICCV09Dataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')

    PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

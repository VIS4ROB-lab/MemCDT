# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from mmseg.datasets.builder import DATASETS

from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(ACDCDataset, self).__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)

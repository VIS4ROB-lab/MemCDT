# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from mmseg.datasets.builder import DATASETS

from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class GTADataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(GTADataset, self).__init__(
            img_suffix='.png', seg_map_suffix='_labelTrainIds.png', **kwargs)

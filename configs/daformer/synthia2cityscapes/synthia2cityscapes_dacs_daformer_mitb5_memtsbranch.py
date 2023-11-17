# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

_base_ = ['./synthia2cityscapes_dacs_daformer_mitb5.py']

model = dict(backbone=dict(type='memcd_mit_b5'), ts_branch=True)

custom_imports = dict(
    imports=[
        'mmseg_uda.datasets.uda_dataset', 'mmseg_uda.datasets.synthia',
        'mmseg_uda.models.uda.dacs',
        'mmseg_uda.models.backbones.memcd_mix_transformer',
        'mmseg_uda.models.decode_heads.daformer_head'
    ],
    allow_failed_imports=False)

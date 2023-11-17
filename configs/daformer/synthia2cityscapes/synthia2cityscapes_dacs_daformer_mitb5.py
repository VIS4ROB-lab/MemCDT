# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# The configuration for UDA training is based on:
# https://github.com/lhoyer/DAFormer
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer.
# Licensed under the Apache License, Version 2.0.

_base_ = [
    '../../_base_/models/daformer_sepaspp_mitb5.py',
    '../../_base_/datasets/uda_synthia2cityscapes_512x512.py',
    '../../_base_/schedules/optimizers/adamw.py',
    '../../_base_/schedules/policies/poly10warm.py',
    '../../_base_/schedules/schedule_40k.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='DACS',
    decode_head=dict(num_classes=19),
    # DACS parameters
    ema_alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75)

data = dict(
    train=dict(
        rcs_cfg=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)),
    samples_per_gpu=2,
    workers_per_gpu=4)

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = None

custom_imports = dict(
    imports=[
        'mmseg_uda.datasets.uda_dataset', 'mmseg_uda.datasets.synthia',
        'mmseg_uda.models.uda.dacs',
        'mmseg_uda.models.backbones.mix_transformer',
        'mmseg_uda.models.decode_heads.daformer_head'
    ],
    allow_failed_imports=False)

# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# The Rare Class Sampling (RCS) strategy is based on:
# https://github.com/lhoyer/DAFormer
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer.
# Licensed under the Apache License, Version 2.0.

import json
import os.path as osp

import mmcv
import numpy as np
import torch
from mmseg.datasets.builder import DATASETS, build_dataset


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, rcs_cfg=None):
        self.source = build_dataset(source)
        self.target = build_dataset(target)

        self.CLASSES = self.target.CLASSES
        self.PALETTE = self.target.PALETTE
        self.ignore_index = self.target.ignore_index
        assert self.CLASSES == self.source.CLASSES
        assert self.PALETTE == self.source.PALETTE
        assert self.ignore_index == self.source.ignore_index

        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                source['data_root'], self.rcs_class_temp)
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(source['data_root'], 'samples_with_class.json'),
                    'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f = np.random.choice(self.samples_with_class[c])
        i = self.file_to_idx[f]
        s = self.source[i]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s = self.source[i]
        return s, i

    def __getitem__(self, idx):
        if self.rcs_enabled:
            sample_s, _ = self.get_rare_class_sample()
            # NOTE: This mimics how target sampling in DAFormer RCS works.
            # Could simply use self.target[idx % len(self.target)] here.
            idx_t = np.random.choice(range(len(self.target)))
            sample_t = self.target[idx_t]
        else:
            sample_s = self.source[idx // len(self.target)]
            sample_t = self.target[idx % len(self.target)]

        return {
            **sample_s, 'img_t': sample_t['img'],
            'img_metas_t': sample_t['img_metas']
        }

    def __len__(self):
        return len(self.source) * len(self.target)

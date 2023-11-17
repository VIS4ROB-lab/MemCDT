# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# The baseline UDA pipeline (excluding the training of the cross-domain
# branches) is based on:
# https://github.com/lhoyer/DAFormer
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer.
# Licensed under the Apache License, Version 2.0.

# The domain-mixing strategy (also employed in DAFormer) is based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.

import random
from copy import deepcopy

import numpy as np
import torch

from mmseg.core import add_prefix
from mmseg.models.builder import SEGMENTORS, build_segmentor

from mmseg_uda.utils.dacs_transforms import (get_class_masks, get_mean_std,
                                             strong_transform)
from mmseg_uda.utils.utils import downscale_label_ratio
from .uda_pipeline import UDAPipeline, get_module


@SEGMENTORS.register_module()
class DACS(UDAPipeline):

    def __init__(self,
                 ema_alpha=0.99,
                 pseudo_threshold=0.968,
                 pseudo_weight_ignore_top=0,
                 pseudo_weight_ignore_bottom=0,
                 blur=True,
                 color_jitter_strength=0.2,
                 color_jitter_probability=0.2,
                 imnet_feature_dist_lambda=0,
                 imnet_feature_dist_classes=None,
                 imnet_feature_dist_scale_min_ratio=0.75,
                 st_branch=False,
                 ts_branch=False,
                 **encoder_decoder_cfg):
        super(DACS, self).__init__(ema_alpha=ema_alpha, **encoder_decoder_cfg)

        # Pseudo-labels
        self.pseudo_threshold = pseudo_threshold
        self.psweight_ignore_top = pseudo_weight_ignore_top
        self.psweight_ignore_bottom = pseudo_weight_ignore_bottom

        # Mixing
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability

        # ImageNet feature distance
        self.imnet_fdist_enabled = imnet_feature_dist_lambda > 0
        if self.imnet_fdist_enabled:
            imnet_cfg = deepcopy(encoder_decoder_cfg)
            imnet_cfg['type'] = 'EncoderDecoder'
            self.imnet_model = build_segmentor(imnet_cfg)
            self.fdist_lambda = imnet_feature_dist_lambda
            self.fdist_classes = imnet_feature_dist_classes
            self.fdist_scale_min_ratio = imnet_feature_dist_scale_min_ratio

        # Cross-domain branches
        self.st_branch_enabled = st_branch
        self.ts_branch_enabled = ts_branch

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.imnet_fdist_enabled
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            # self.debug_fdist_mask = fdist_mask
            # self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        # feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_uda(self, img, img_metas, gt_semantic_seg, img_t, img_metas_t):
        batch_size = img.shape[0]
        device = img.device
        log_vars = dict(loss=0)

        # Train on source images
        loss_s = self.model_forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        feat_s = loss_s.pop('features')
        loss_s = add_prefix(loss_s, 's')
        loss_s, log_vars_s = self._parse_losses(loss_s)
        log_vars_s['loss'] += log_vars['loss']
        log_vars.update(log_vars_s)
        loss_s.backward(retain_graph=self.imnet_fdist_enabled)

        # ImageNet feature distance
        if self.imnet_fdist_enabled:
            loss_feat, log_vars_feat = self.calc_feat_dist(
                img, gt_semantic_seg, feat_s)
            log_vars_feat['loss'] += log_vars['loss']
            log_vars.update(log_vars_feat)
            loss_feat.backward()

        # Generate pseudo-labels
        pseudo_label, pseudo_prob = self.generate_ema_pseudo_labels(
            img_t, img_metas_t)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=device)

        # Don't trust pseudo-labels in regions with potential rectification
        # artifacts. This can lead to a pseudo-label drift from sky towards
        # building or traffic light.
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0

        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=device)

        # Apply cross-domain mixing
        means, stds = get_mean_std(img_metas, device)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], img_t[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        loss_t = self.model_forward_train(mixed_img, img_metas_t, mixed_lbl,
                                          pseudo_weight)
        loss_t = add_prefix(loss_t, 't')
        loss_t, log_vars_t = self._parse_losses(loss_t)
        log_vars_t['loss'] += log_vars['loss']
        log_vars.update(log_vars_t)
        loss_t.backward()

        # Train cross-domain branches
        if self.st_branch_enabled:
            loss_st = self.model_forward_train((img, mixed_img), img_metas,
                                               gt_semantic_seg)
            loss_st = add_prefix(loss_st, 'st')
            loss_st, log_vars_st = self._parse_losses(loss_st)
            log_vars_st['loss'] += log_vars['loss']
            log_vars.update(log_vars_st)
            loss_st.backward()

        if self.ts_branch_enabled:
            loss_ts = self.model_forward_train((mixed_img, img), img_metas_t,
                                               mixed_lbl, pseudo_weight)
            loss_ts = add_prefix(loss_ts, 'ts')
            loss_ts, log_vars_ts = self._parse_losses(loss_ts)
            log_vars_ts['loss'] += log_vars['loss']
            log_vars.update(log_vars_ts)
            loss_ts.backward()

        return log_vars

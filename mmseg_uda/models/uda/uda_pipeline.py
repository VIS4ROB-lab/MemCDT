# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# The EMA model update is based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.

import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import force_fp32
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

import mmseg.version
from mmseg.core import add_prefix
from mmseg.models.builder import build_segmentor
from mmseg.models.losses import accuracy
from mmseg.models.segmentors import EncoderDecoder
from mmseg.ops import resize


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.
    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.
    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


def head_forward_train(head,
                       x,
                       seg_labels,
                       seg_weights=None,
                       return_seg_logits=False):
    seg_logits = head(x)
    loss_decode = compute_decode_loss(
        head.loss_decode,
        seg_logits,
        seg_labels,
        seg_weight=seg_weights,
        ignore_index=head.ignore_index,
        sampler=head.sampler,
        align_corners=head.align_corners,
        return_seg_logits=return_seg_logits)
    return loss_decode


@force_fp32(apply_to=('seg_logit', ))
def compute_decode_loss(loss_decode,
                        seg_logit,
                        seg_label,
                        seg_weight=None,
                        ignore_index=255,
                        sampler=None,
                        align_corners=False,
                        return_seg_logits=False):
    """Compute segmentation loss."""
    losses = dict()
    seg_logit = resize(
        input=seg_logit,
        size=seg_label.shape[2:],
        mode='bilinear',
        align_corners=align_corners)
    if return_seg_logits:
        losses['seg_logits'] = seg_logit
    if sampler is not None:
        sampler_weight = sampler.sample(seg_logit, seg_label)
        if seg_weight is not None:
            seg_weight = seg_weight * sampler_weight
        else:
            seg_weight = sampler_weight

    seg_label = seg_label.squeeze(1)

    if (mmseg.version.version_info >=
            mmseg.version.parse_version_info('0.18.0')):
        if not isinstance(loss_decode, nn.ModuleList):
            losses_decode = [loss_decode]
        else:
            losses_decode = loss_decode
        for loss in losses_decode:
            if loss.loss_name not in losses:
                losses[loss.loss_name] = loss(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=ignore_index)
            else:
                losses[loss.loss_name] += loss(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=ignore_index)
    else:
        losses['loss_seg'] = loss_decode(
            seg_logit, seg_label, weight=seg_weight, ignore_index=ignore_index)

    if (mmseg.version.version_info >=
            mmseg.version.parse_version_info('0.22.0')):
        losses['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=ignore_index)
    else:
        losses['acc_seg'] = accuracy(seg_logit, seg_label)

    return losses


class UDAPipeline(EncoderDecoder, metaclass=ABCMeta):

    def __init__(self, ema_alpha=0, **encoder_decoder_cfg):
        super(UDAPipeline, self).__init__(**encoder_decoder_cfg)

        # Init EMA model (i.e. teacher) if needed
        self.ema_model_enabled = ema_alpha > 0
        if self.ema_model_enabled:
            ema_cfg = deepcopy(encoder_decoder_cfg)
            ema_cfg['type'] = 'EncoderDecoder'
            self.ema_model = build_segmentor(ema_cfg)
            self.ema_alpha = ema_alpha

        self.local_iter = 0

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_ema_model_params(self):
        return self.get_ema_model().parameters()

    def get_model_params(self):
        for name, param in self.named_parameters():
            if '_model' not in name:
                yield param

    def _init_ema_weights(self):
        for param in self.get_ema_model_params():
            param.detach_()
        mp = list(self.get_model_params())
        mcp = list(self.get_ema_model_params())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.ema_alpha)
        for ema_param, param in zip(self.get_ema_model_params(),
                                    self.get_model_params()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def generate_ema_pseudo_labels(self, img, img_metas, return_feat=False):
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(img, img_metas)
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        return pseudo_label, pseudo_prob

    @abstractmethod
    def forward_uda(self, img, img_metas, gt_semantic_seg, img_t, img_metas_t,
                    **kwargs):
        """Placeholder for Forward function for UDA training."""
        pass

    def forward_train(self, img, img_metas, gt_semantic_seg, img_t,
                      img_metas_t, **kwargs):

        # Init/update EMA model
        if self.ema_model_enabled:
            if self.local_iter == 0:
                self._init_ema_weights()
                # assert params_equal(self.get_ema_model(), self)
            elif self.local_iter > 0:
                self._update_ema(self.local_iter)
                # assert not _params_equal(self.get_ema_model(), self)
                # assert self.get_ema_model().training

        losses = self.forward_uda(img, img_metas, gt_semantic_seg, img_t,
                                  img_metas_t, **kwargs)
        self.local_iter += 1

        return losses

    def model_forward_train(self,
                            img,
                            img_metas,
                            seg_labels,
                            seg_weights=None,
                            return_feat=False,
                            return_seg_logits=False):
        x = self.extract_feat(img)

        losses = dict()

        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, img_metas, seg_labels,
                                                      seg_weights,
                                                      return_seg_logits)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, seg_labels, seg_weights)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   seg_labels,
                                   seg_weights=None,
                                   return_seg_logits=False):
        losses = dict()
        loss_decode = head_forward_train(self.decode_head, x, seg_labels,
                                         seg_weights, return_seg_logits)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      seg_labels,
                                      seg_weights=None):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = head_forward_train(aux_head, x, seg_labels,
                                              seg_weights)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = head_forward_train(self.auxiliary_head, x, seg_labels,
                                          seg_weights)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        # log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))

        return outputs

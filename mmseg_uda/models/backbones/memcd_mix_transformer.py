# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# The architecture is based on the MixVisionTransformer implementation in:
# https://github.com/NVlabs/SegFormer
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# Licensed under the NVIDIA Source Code License

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

from .mix_transformer import Mlp, OverlapPatchEmbed


class MemCDAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward_standard(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_crossdomain(self, xs, xt, xts, H, W):
        B, N, C = xs.shape

        qts = self.q(xts).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1,
                                                               3).contiguous()
        with torch.no_grad():
            qs = self.q(xs).reshape(B, N, self.num_heads, C //
                                    self.num_heads).permute(0, 2, 1,
                                                            3).contiguous()
            qt = self.q(xt).reshape(B, N, self.num_heads, C //
                                    self.num_heads).permute(0, 2, 1,
                                                            3).contiguous()

        if self.sr_ratio > 1:
            xts_ = xts.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            xts_ = self.sr(xts_).reshape(B, C, -1).permute(0, 2,
                                                           1).contiguous()
            xts_ = self.norm(xts_)
            kvts = self.kv(xts_).reshape(B, -1, 2, self.num_heads,
                                         C // self.num_heads).permute(
                                             2, 0, 3, 1, 4).contiguous()
            with torch.no_grad():
                xs_ = xs.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
                xs_ = self.sr(xs_).reshape(B, C, -1).permute(0, 2,
                                                             1).contiguous()
                xs_ = self.norm(xs_)
                kvs = self.kv(xs_).reshape(B, -1, 2, self.num_heads,
                                           C // self.num_heads).permute(
                                               2, 0, 3, 1, 4).contiguous()
                xt_ = xt.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
                xt_ = self.sr(xt_).reshape(B, C, -1).permute(0, 2,
                                                             1).contiguous()
                xt_ = self.norm(xt_)
                kvt = self.kv(xt_).reshape(B, -1, 2, self.num_heads,
                                           C // self.num_heads).permute(
                                               2, 0, 3, 1, 4).contiguous()

        else:
            kvts = self.kv(xts).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                                            2, 0, 3, 1, 4).contiguous()
            with torch.no_grad():
                kvs = self.kv(xs).reshape(B, -1, 2, self.num_heads,
                                          C // self.num_heads).permute(
                                              2, 0, 3, 1, 4).contiguous()
                kvt = self.kv(xt).reshape(B, -1, 2, self.num_heads,
                                          C // self.num_heads).permute(
                                              2, 0, 3, 1, 4).contiguous()

        ks, vs = kvs[0], kvs[1]
        kt, vt = kvt[0], kvt[1]
        kts, vts = kvts[0], kvts[1]

        attn_ts1 = (qts @ kts.transpose(-2, -1).contiguous()) * self.scale
        attn_ts1 = attn_ts1.softmax(dim=-1)
        attn_ts1 = self.attn_drop(attn_ts1)
        with torch.no_grad():
            attn_s = (qs @ ks.transpose(-2, -1).contiguous()) * self.scale
            attn_s = attn_s.softmax(dim=-1)
            attn_s = self.attn_drop(attn_s)
            attn_t = (qt @ kt.transpose(-2, -1).contiguous()) * self.scale
            attn_t = attn_t.softmax(dim=-1)
            attn_t = self.attn_drop(attn_t)
            attn_ts2 = (qt @ ks.transpose(-2, -1).contiguous()) * self.scale
            attn_ts2 = attn_ts2.softmax(dim=-1)
            attn_ts2 = self.attn_drop(attn_ts2)

        xts1 = (attn_ts1 @ vts).transpose(1, 2).contiguous().reshape(B, N, C)
        with torch.no_grad():
            xs = (attn_s @ vs).transpose(1, 2).contiguous().reshape(B, N, C)
            xt = (attn_t @ vt).transpose(1, 2).contiguous().reshape(B, N, C)
            xts2 = (attn_ts2 @ vs).transpose(1,
                                             2).contiguous().reshape(B, N, C)

        xts = 0.5 * (xts1 + xts2.detach())
        xts = self.proj(xts)
        xts = self.proj_drop(xts)
        with torch.no_grad():
            xs = self.proj(xs)
            xs = self.proj_drop(xs)
            xt = self.proj(xt)
            xt = self.proj_drop(xt)

        return xs, xt, xts

    def forward(self, x, H, W):
        if isinstance(x, tuple):
            return self.forward_crossdomain(x[0], x[1], x[2], H, W)
        else:
            return self.forward_standard(x, H, W)


class MemCDBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MemCDAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward_standard(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def forward_crossdomain(self, x, H, W):
        xts = self.norm1(x[2])
        with torch.no_grad():
            xs = self.norm1(x[0])
            xt = self.norm1(x[1])
        xs, xt, xts = self.attn((xs, xt, xts), H, W)
        xts = x[2] + self.drop_path(xts)
        with torch.no_grad():
            xs = x[0] + self.drop_path(xs)
            xt = x[1] + self.drop_path(xt)
        xts = xts + self.drop_path(self.mlp(self.norm2(xts), H, W))
        with torch.no_grad():
            xs = xs + self.drop_path(self.mlp(self.norm2(xs), H, W))
            xt = xt + self.drop_path(self.mlp(self.norm2(xt), H, W))

        return xs, xt, xts

    def forward(self, x, H, W):
        if isinstance(x, tuple):
            return self.forward_crossdomain(x, H, W)
        else:
            return self.forward_standard(x, H, W)


@BACKBONES.register_module()
class MemCDMixVisionTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            MemCDBlock(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            MemCDBlock(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            MemCDBlock(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            MemCDBlock(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_standard(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward_crossdomain(self, x):
        B = x[0].shape[0]
        outs = []

        # stage 1
        xts, H, W = self.patch_embed1(x[0])
        with torch.no_grad():
            xs, _, _ = self.patch_embed1(x[1])
            xt, _, _ = self.patch_embed1(x[0])
        for i, blk in enumerate(self.block1):
            xs, xt, xts = blk((xs, xt, xts), H, W)
        xts = self.norm1(xts)
        xts = xts.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            xs = self.norm1(xs)
            xs = xs.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            xt = self.norm1(xt)
            xt = xt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(xts)

        # stage 2
        xts, H, W = self.patch_embed2(xts)
        with torch.no_grad():
            xs, _, _ = self.patch_embed2(xs)
            xt, _, _ = self.patch_embed2(xt)
        for i, blk in enumerate(self.block2):
            xs, xt, xts = blk((xs, xt, xts), H, W)
        xts = self.norm2(xts)
        xts = xts.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            xs = self.norm2(xs)
            xs = xs.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            xt = self.norm2(xt)
            xt = xt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(xts)

        # stage 3
        xts, H, W = self.patch_embed3(xts)
        with torch.no_grad():
            xs, _, _ = self.patch_embed3(xs)
            xt, _, _ = self.patch_embed3(xt)
        for i, blk in enumerate(self.block3):
            xs, xt, xts = blk((xs, xt, xts), H, W)
        xts = self.norm3(xts)
        xts = xts.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            xs = self.norm3(xs)
            xs = xs.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            xt = self.norm3(xt)
            xt = xt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(xts)

        # stage 4
        xts, H, W = self.patch_embed4(xts)
        with torch.no_grad():
            xs, _, _ = self.patch_embed4(xs)
            xt, _, _ = self.patch_embed4(xt)
        for i, blk in enumerate(self.block4):
            xs, xt, xts = blk((xs, xt, xts), H, W)
        xts = self.norm4(xts)
        xts = xts.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(xts)

        return outs

    def forward(self, x):
        if isinstance(x, tuple):
            return self.forward_crossdomain(x)
        else:
            return self.forward_standard(x)


@BACKBONES.register_module()
class memcd_mit_b0(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class memcd_mit_b1(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class memcd_mit_b2(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class memcd_mit_b3(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class memcd_mit_b4(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class memcd_mit_b5(MemCDMixVisionTransformer):

    def __init__(self, **kwargs):
        super(memcd_mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)

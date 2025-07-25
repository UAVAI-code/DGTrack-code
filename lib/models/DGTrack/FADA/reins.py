import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import reduce

from operator import mul

from torch import Tensor
import numpy as np



class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 60,
        use_softmax: bool = True,
        link_token_to_query: bool = False,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,

        use_lora: bool = True,
        lora_dim: int = 8,

    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f

        self.use_lora = use_lora
        self.lora_dim = lora_dim

        self.create_model()


    def create_model(self):

        if self.use_lora:
            self.learnable_tokens_a = nn.Parameter(
                torch.empty([self.num_layers, self.token_length, self.lora_dim])
            )
            self.learnable_tokens_b = nn.Parameter(
                torch.empty([self.num_layers, self.lora_dim, int(self.embed_dims / 4)])
            )
        else:
            self.learnable_tokens = nn.Parameter(
                torch.empty([self.num_layers, self.token_length, int(self.embed_dims/4)])
            )

        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(int(self.embed_dims/4), int(self.embed_dims/4))
        self.mlp_delta_f = nn.Linear(int(self.embed_dims/4), int(self.embed_dims/4))

        if self.use_lora:
            self.learnable_tokens_a2 = nn.Parameter(
                torch.empty([self.num_layers, self.token_length, self.lora_dim])
            )
            self.learnable_tokens_b2 = nn.Parameter(
                torch.empty([self.num_layers, self.lora_dim, int(self.embed_dims / 4) * 3])
            )
        else:
            self.learnable_tokens2 = nn.Parameter(
                torch.empty([self.num_layers, self.token_length, int(self.embed_dims/4)*3])
            )

        self.mlp_token2feat2 = nn.Linear(int(self.embed_dims/4)*3, int(self.embed_dims/4)*3)
        self.mlp_delta_f2 = nn.Linear(int(self.embed_dims/4)*3, int(self.embed_dims/4)*3)

        self.wavepool = WavePool(192)

        if self.use_lora:
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                    + (self.embed_dims * self.lora_dim) ** 0.5
                )
            )
            nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
            nn.init.uniform_(self.learnable_tokens_b.data, -val, val)
        else:
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, (self.patch_size, self.patch_size), 1) + int(self.embed_dims/4)
                )
            )
            nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))


        if self.use_lora:
            val2 = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                    + (int(self.embed_dims / 4) * 3 * self.lora_dim) ** 0.5
                )
            )
            nn.init.uniform_(self.learnable_tokens_a2.data, -val2, val2)
            nn.init.uniform_(self.learnable_tokens_b2.data, -val2, val2)
        else:
            val2 = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, (self.patch_size, self.patch_size), 1) + int(self.embed_dims/4)*3
                )
            )
            nn.init.uniform_(self.learnable_tokens2.data, -val2, val2)
        nn.init.kaiming_uniform_(self.mlp_delta_f2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat2.weight, a=math.sqrt(5))

        self.transform = nn.Linear(int(self.embed_dims/4), self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)

        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

            nn.init.zeros_(self.mlp_delta_f2.weight)
            nn.init.zeros_(self.mlp_delta_f2.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)

            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if self.use_lora:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
        else:
            return self.learnable_tokens[layer]

    def get_tokens2(self, layer: int) -> Tensor:
        if self.use_lora:
            return self.learnable_tokens_a2[layer] @ self.learnable_tokens_b2[layer]
        else:
            return self.learnable_tokens2[layer]

    def forward(
        self,
            idx,
            feats: Tensor,
            layer: int,
            batch_first=False,
            has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)




        len_z = 64
        len_x = 256
        total_len = feats.shape[0]

        feats_z = feats[:len_z, :, :]
        feats_x = feats[len_z:, :, :]

        N_z, B_z, C_z = feats_z.shape
        h_z = int(math.sqrt(len_z))
        w_z = h_z
        feats_z_2d = feats_z.reshape(B_z, C_z, h_z, w_z)

        ll, lh, hl, hh = self.wavepool(feats_z_2d)

        c_z = int(C_z / 4)
        ll, lh, hl, hh = ll.reshape(B_z, N_z, c_z), lh.reshape(B_z, N_z, c_z), hl.reshape(B_z, N_z, c_z), hh.reshape(B_z, N_z, c_z)

        feats_c_z = ll.reshape(N_z, B_z, c_z)
        feats_s_z = torch.cat((lh, hl, hh), -1).reshape(N_z, B_z, c_z * 3)

        feats_z = feats_z_2d.reshape(N_z, B_z, C_z)


        N_x, B_x, C_x = feats_x.shape
        h_x = int(math.sqrt(len_x))
        w_x = h_x
        feats_x_2d = feats_x.reshape(B_x, C_x, h_x, w_x)

        ll, lh, hl, hh = self.wavepool(feats_x_2d)

        c_x = int(C_x / 4)
        ll, lh, hl, hh = ll.reshape(B_x, N_x, c_x), lh.reshape(B_x, N_x, c_x), hl.reshape(B_x, N_x, c_x), hh.reshape(B_x, N_x, c_x)

        feats_c_x = ll.reshape(N_x, B_x, c_x)
        feats_s_x = torch.cat((lh, hl, hh), -1).reshape(N_x, B_x, c_x * 3)

        feats_x = feats_x_2d.reshape(N_x, B_x, C_x)


        feats = torch.cat([feats_z, feats_x], dim=0)

        feats_c = torch.cat([feats_c_z, feats_c_x], dim=0)

        feats_s = torch.cat([feats_s_z, feats_s_x], dim=0)


        tokens1 = self.get_tokens(layer)
        tokens2 = self.get_tokens2(layer)


        delta_featc = self.forward_delta_feat(
            feats_c,
            tokens1,
            layer,
        )

        if len(idx) > 0:
            delta_feats = feats_s
        else:
            delta_feats = self.forward_delta_feat2(
                feats_s,
                tokens2,
                layer,
            )


        delta_feat = torch.cat((delta_featc, delta_feats), -1)
        



        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)

        if batch_first:
            feats = feats.permute(1, 0, 2)

        return feats


    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:

        attn = torch.einsum("nbc,mc->nbm", feats, tokens)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )

        
        delta_f = self.mlp_delta_f(delta_f + feats)

        return delta_f



        



    def forward_delta_feat2(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:

        attn = torch.einsum("nbc,mc->nbm", feats, tokens)

        attn = F.instance_norm(attn)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat2(tokens[1:, :]),
        )

        delta_f = self.mlp_delta_f2(delta_f + feats)
        return delta_f





def pcnorm(residual):
    norm = F.instance_norm(residual)

    phase, _ = decompose(residual, 'phase')
    _, norm_amp = decompose(norm, 'amp')

    residual = compose(
        phase,
        norm_amp
    )
    
    return residual

def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp

def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, s=x.shape[1:], norm='ortho')

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)



# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""ConvolutionModule definition."""

from typing import Tuple

import torch
from torch import nn


class ConvolutionModule(nn.Module):
    """ConvolutionModule in ChunkFormer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True,
                 use_dynamic_conv: bool = False):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()
        self.use_dynamic_conv = use_dynamic_conv
        self.channels = channels
        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        elif use_dynamic_conv:
            assert (kernel_size - 1) % 2 == 0
            padding = 0
            self.lorder = (kernel_size - 1)//2
        else:
            # kernel_size should be an odd number for none causal convolution
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        
    def forward_parallel_chunk(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
        truncated_context_size: int = 0

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)
        lorder = self.kernel_size//2
        chunk_size = x.shape[-1]
        if cache.size(0) == 0:
            cache = torch.zeros(self.channels, lorder).to(x.device)
        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        #----------Overlapping Chunk Transformation-----------------------------------
        x = x.transpose(0, 1).reshape( self.channels, -1)  # [C, n_chunk * T]
        x = torch.cat([cache, x], dim=-1)
        new_cache = x[:, :truncated_context_size + cache.size(-1)][:, -cache.size(-1):].cpu()
        x = nn.functional.pad(x, (0, lorder), 'constant', 0.0)
        x = x.unfold(-1, chunk_size + 2 * lorder, chunk_size).transpose(0, 1) #[n_chunk +1, C, cnn_cache_size]
        #-----------------------------------------------------------------------------

        if mask_pad.size(2) > 0:  # time > 0
            x = torch.where(mask_pad, x, 0)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            # x.masked_fill_(~mask_pad[:, :, self.lorder:], 0.0)
            x.masked_fill_(~mask_pad[:, :, self.lorder:-self.lorder], 0.0)

        return x.transpose(1, 2), new_cache

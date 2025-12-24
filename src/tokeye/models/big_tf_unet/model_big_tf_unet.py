import torch
import torch.nn as nn

from ...models.tokeye.model_tokeye import (
    TokEyeConvBlock,
    TokEyeDownBlock,
    TokEyeUpBlock,
)

from .config_big_tf_unet import BigTFUNetConfig

class BigTFUNetModel(nn.Module):

    def __init__(self, config: BigTFUNetConfig):
        super().__init__()
        self.config = config
        
        # Layer sizes
        layer_sizes: list[int] = [
            config.first_layer_size * 2**i 
            for i in range(config.num_layers)
        ]

        # Initial Channel Convolution
        self.in_conv = TokEyeConvBlock(
            config.in_channels,
            layer_sizes[0],
            dropout_rate=config.dropout_rate,
        )

        # Encoder
        encoder: list[TokEyeDownBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i + 1]
            encoder.append(TokEyeDownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout_rate=config.dropout_rate,
            ))
        self.encoder = nn.ModuleList(encoder)

        # Decoder
        decoder: list[TokEyeUpBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[-i - 1]
            out_ch = layer_sizes[-i - 2]
            decoder.append(TokEyeUpBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout_rate=config.dropout_rate,
            ))
        self.decoder = nn.ModuleList(decoder)

        # Final Channel Convolution
        self.out_conv = nn.Conv2d(
            layer_sizes[0],
            config.out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        input_BCHW: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        skip_BCHW: list[torch.Tensor] = []

        # Channel Convolution
        encode_BCHW = self.in_conv(input_BCHW)
        skip_BCHW.append(encode_BCHW)

        # Encoder
        for layer in self.encoder:
            encode_BCHW = layer(encode_BCHW)
            skip_BCHW.append(encode_BCHW)

        # Bottleneck
        decode_BCHW = encode_BCHW

        # Decoder
        for i, layer in enumerate(self.decoder):
            skip_idx = len(skip_BCHW) - i - 2
            decode_BCHW = layer(
                decode_BCHW,
                skip_BCHW[skip_idx],
            )

        # Channel Convolution
        output_BCHW = self.out_conv(decode_BCHW)

        return (output_BCHW,)
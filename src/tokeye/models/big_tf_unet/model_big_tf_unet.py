import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_big_tf_unet import BigTFUNetConfig


class BigTFUNetConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout_rate: float = 0.0,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers: list[nn.Module] = []
        
        layers.extend([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            ])

        if dropout_rate > 0:
            layers.extend([nn.Dropout2d(p=dropout_rate)])

        layers.extend([
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            ])

        if dropout_rate > 0:
            layers.extend([nn.Dropout2d(p=dropout_rate)])

        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(hidden_states)


class BigTFUNetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            BigTFUNetConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.down(hidden_states)


class BigTFUNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=kernel_size,
            mode="bilinear",
            align_corners=True,
        )
        self.conv = BigTFUNetConvBlock(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        hidden_states_1: torch.Tensor,
        hidden_states_2: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states_1 = self.up(hidden_states_1)

        diffY = hidden_states_2.size()[2] - hidden_states_1.size()[2]
        diffX = hidden_states_2.size()[3] - hidden_states_1.size()[3]

        hidden_states_1 = F.pad(
            hidden_states_1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        hidden_states = torch.cat([hidden_states_2, hidden_states_1], dim=1)
        return self.conv(hidden_states)


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
        self.in_conv = BigTFUNetConvBlock(
            config.in_channels,
            layer_sizes[0],
            dropout_rate=config.dropout_rate,
        )

        # Encoder
        encoder: list[BigTFUNetDownBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[i]
            out_ch = layer_sizes[i + 1]
            encoder.append(BigTFUNetDownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                dropout_rate=config.dropout_rate,
            ))
        self.encoder = nn.ModuleList(encoder)

        # Decoder
        decoder: list[BigTFUNetUpBlock] = []
        for i in range(config.num_layers - 1):
            in_ch = layer_sizes[-i - 1]
            out_ch = layer_sizes[-i - 2]
            decoder.append(BigTFUNetUpBlock(
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
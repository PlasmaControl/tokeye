import torch
import torch.nn as nn
import torch.nn.functional as F


class TokEyeConvBlock(nn.Module):
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


class TokEyeDownBlock(nn.Module):
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
            TokEyeConvBlock(
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


class TokEyeUpBlock(nn.Module):
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
        self.conv = TokEyeConvBlock(
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


if __name__ == "__main__":
    # python -m tokeye.models.modules.nn
    conv_model = TokEyeConvBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(conv_model(x).shape)

    down_model = TokEyeDownBlock(3, 64)
    x = torch.randn(1, 3, 256, 256)
    print(down_model(x).shape)

    up_model = TokEyeUpBlock(3, 64)
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 64, 256, 256)
    print(up_model(x1, x2).shape)

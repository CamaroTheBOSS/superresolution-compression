from dataclasses import dataclass
from typing import Optional, Any, List

import torch
import torch.nn.functional as f
from torch import Tensor
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d


@dataclass
class VSRModelOutput:
    reconstruction: Optional[Tensor] = None
    loss: Optional[dict] = None


class VSRModel(Module):
    def __init__(self):
        super(VSRModel, self).__init__()
        # Encoder
        self.Conv2d_A_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_B_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_C_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_A = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Conv2d_D_1x1 = BasicConv2d(64, 196, kernel_size=3)
        self.Conv2d_E_3x3 = BasicConv2d(196, 256, kernel_size=3, padding=1)
        self.MaxPool_B = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.BranchBlockA1 = BranchBlockA(256, 32)
        self.BranchBlockA2 = BranchBlockA(416, 64)
        self.BranchBlockA3 = BranchBlockA(448, 128)

        self.BranchBlockB1 = BranchBlockB(512)

        # Decoder
        self.Conv2d_Encoder_A = BasicConv2d(672, 672, kernel_size=3, padding=1)
        self.Conv2d_Encoder_B = BasicConv2d(672, 672, kernel_size=3, padding=1)

        self.DeConv2d_A = ConvTranspose2d(672, 320, kernel_size=3, stride=3, padding=1)

        self.Conv2d_Encoder_C = BasicConv2d(320, 320, kernel_size=(1, 3), padding=(0, 1))
        self.Conv2d_Encoder_D = BasicConv2d(320, 320, kernel_size=(3, 1), padding=(1, 0))

        self.DeConv2d_B = ConvTranspose2d(320, 150, kernel_size=(3, 3), stride=3, padding=1)

        self.Conv2d_Encoder_E = BasicConv2d(150, 150, kernel_size=(3, 3), padding=1)
        self.Conv2d_Encoder_F = BasicConv2d(150, 150, kernel_size=(3, 3), padding=1)

        self.DeConv2d_C = ConvTranspose2d(150, 70, kernel_size=(3, 3), stride=3, padding=1)

        self.Conv2d_Encoder_G = BasicConv2d(70, 70, kernel_size=(1, 1))
        self.Conv2d_Encoder_H = BasicConv2d(70, 70, kernel_size=(1, 1))

        self.DeConv2d_D = ConvTranspose2d(70, 35, kernel_size=(3, 3), stride=3, padding=1)

        self.Conv2d_Out = BasicConv2d(35, 3, kernel_size=(1, 1))

    def encode(self, frames):
        # 3 x H x W
        out = self.Conv2d_A_3x3(frames)
        # 32 x H//2 x W//2
        out = self.Conv2d_B_3x3(out)
        # 32 x H//2 x W//2
        out = self.Conv2d_C_3x3(out)
        # 64 x H//2 x W//2
        out = self.MaxPool_A(out)

        # 32 x H//4 x W//4
        out = self.Conv2d_D_1x1(out)
        # 196 x H//4-2 x W//4-2
        out = self.Conv2d_E_3x3(out)
        # 256 x H//4-2 x W//4-2
        out = self.MaxPool_B(out)

        # 256 x (H//4-2)//2 x (W//4-2)//2
        out = self.BranchBlockA1(out)
        # 416 x (H//4-2)//2 x (W//4-2)//2
        out = self.BranchBlockA2(out)
        # 448 x (H//4-2)//2 x (W//4-2)//2
        out = self.BranchBlockA3(out)
        # 512 x (H//4-2)//2 x (W//4-2)//2
        out = self.BranchBlockB1(out)
        # 672 x (H//4-2)//4 + 1 x (W//4-2)//4
        return out

    def decode(self, features: Tensor) -> Tensor:
        # 672 x (H//4-2)//4 + 1 x (W//4-2)//4
        out = self.Conv2d_Encoder_A(features)
        # 672 x (H//4-2)//4 + 1 x (W//4-2)//4
        out = self.Conv2d_Encoder_B(out)
        # 672 x (H//4-2)//4 + 1 x (W//4-2)//4
        out = self.DeConv2d_A(out)
        out = f.relu(out)

        # 672 x (H//4-2)//2 x (W//4-2)//2 - 1
        out = self.Conv2d_Encoder_C(out)
        out = self.Conv2d_Encoder_D(out)
        out = self.DeConv2d_B(out)
        out = f.relu(out)

        out = self.Conv2d_Encoder_E(out)
        out = self.Conv2d_Encoder_F(out)
        out = self.DeConv2d_C(out)
        out = f.relu(out)

        out = self.Conv2d_Encoder_G(out)
        out = self.Conv2d_Encoder_H(out)
        out = self.DeConv2d_D(out)
        out = f.relu(out)

        out = self.Conv2d_Out(out)
        out = f.relu(out)
        return out

    def forward(self, frames: Tensor, labels: Tensor) -> VSRModelOutput:  # input shape: B C H W
        desired_size = (2 * frames.shape[2], 2 * frames.shape[3])
        interpolated_frames = torch.nn.functional.interpolate(frames, size=desired_size,
                                                              mode="bilinear", align_corners=False)
        features = self.encode(interpolated_frames)
        reconstruction = self.decode(features)
        reconstruction = torch.nn.functional.interpolate(reconstruction, size=desired_size,
                                                         mode="bilinear", align_corners=False)
        for rec in reconstruction:
            print(f"{rec.min(), rec.max()}")
        print("----------------------------")
        reconstruction = reconstruction + interpolated_frames

        loss = {}
        if self.training:
            loss["MAE"] = torch.mean(torch.abs(reconstruction - labels))

        return VSRModelOutput(
            reconstruction=reconstruction,
            loss=loss
        )


class BasicConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, bias=True, **kwargs)
        # self.bn = BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # x = self.bn(x)
        return f.relu(x, inplace=True)


class BranchBlockA(Module):
    def __init__(self, in_channels=640, pool_features=32):
        super(BranchBlockA, self).__init__()
        self.Branch_A_1x1 = BasicConv2d(in_channels, 128, kernel_size=3, padding=1)

        self.Branch_B_1x3 = BasicConv2d(in_channels, 64, kernel_size=(1, 3), padding=(0, 1))
        self.Branch_B_3x1 = BasicConv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))

        self.Branch_C_1x1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.Branch_C_5x5 = BasicConv2d(96, 128, kernel_size=5, padding=2)

        self.PoolFeatures = BasicConv2d(in_channels, pool_features, kernel_size=1, padding=1)

    def _forward(self, images) -> List[Tensor]:
        b_A = self.Branch_A_1x1(images)

        b_B = self.Branch_B_1x3(images)
        b_B = self.Branch_B_3x1(b_B)

        b_C = self.Branch_C_1x1(images)
        b_C = self.Branch_C_5x5(b_C)

        pool = f.avg_pool2d(images, kernel_size=3, stride=1)
        pool = self.PoolFeatures(pool)

        outputs = [b_A, b_B, b_C, pool]
        return outputs

    def forward(self, images) -> Tensor:
        output = self._forward(images)
        return torch.cat(output, 1)


class BranchBlockB(Module):
    def __init__(self, in_channels=640):
        super(BranchBlockB, self).__init__()

        self.Branch_A_3x3 = BasicConv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)

        self.Branch_B_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.Branch_B_3x3 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.Branch_B_3x3_2 = BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)

    def _forward(self, images) -> List[Tensor]:
        b_A = self.Branch_A_3x3(images)

        b_B = self.Branch_B_1x1(images)
        b_B = self.Branch_B_3x3(b_B)
        b_B = self.Branch_B_3x3_2(b_B)

        pool = f.max_pool2d(images, kernel_size=3, stride=2, padding=1)
        outputs = [b_A, b_B, pool]
        return outputs

    def forward(self, images) -> Tensor:
        output = self._forward(images)
        return torch.cat(output, 1)

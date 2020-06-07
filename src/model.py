import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv3d, ReLU, BatchNorm3d, Linear, Tanh, Dropout, BatchNorm1d
from torch.nn.modules.flatten import Flatten


class ExpansionBlock(Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.expand = Sequential(Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1), ReLU())
        self.conv = Sequential(Conv3d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, dilation=dilation, groups=mid_channels), ReLU())
        self.bottleneck = Sequential(Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1), BatchNorm3d(out_channels))

    def forward(self, input: Tensor):
        x = self.expand(input)
        x = self.conv(x)
        x = self.bottleneck(x)
        if input.shape == x.shape:
            x = input + x
        return x


class Model(Module):
    def __init__(self):
        super().__init__()
        self.fmri_feature0 = Sequential(
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 16),
            ExpansionBlock(16, 64, 16, dilation=1),
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 8),
            Flatten(), Linear(428400, 32), Tanh()
        )
        self.fmri_feature1 = Sequential(
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 16),
            ExpansionBlock(16, 64, 16, dilation=2),
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 8),
            Flatten(), Linear(363264, 32), Tanh()
        )
        self.fmri_feature2 = Sequential(
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 16),
            ExpansionBlock(16, 64, 16, dilation=4),
            ExpansionBlock(16, 64, 16), ExpansionBlock(16, 64, 8),
            Flatten(), Linear(253344, 32), Tanh()
        )
        self.fmri_feature3 = Sequential(
            ExpansionBlock(16, 64, 16, dilation=1), ExpansionBlock(16, 64, 16, dilation=2), ExpansionBlock(16, 64, 32, dilation=4),
            ExpansionBlock(32, 64, 16), ExpansionBlock(16, 128, 8),
            Flatten(), Linear(207792, 32), Tanh()
        )
        self.fmri_post_seq = Sequential(
            Linear(128, 1024), ReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), ReLU(),
            Linear(512, 512), BatchNorm1d(512), Tanh()
        )
        self.loading_seq = Sequential(
            Linear(53, 512), ReLU(),
            Linear(512, 1024), ReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), ReLU(),
            Linear(512, 512), ReLU(), Dropout(),
            Linear(512, 512), ReLU(), Dropout(),
            Linear(512, 512), BatchNorm1d(512), Tanh()
        )
        self.final_seq = Sequential(
            Linear(1024, 1024), ReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), ReLU(),
            Linear(512, 512), BatchNorm1d(512), ReLU(),
            Linear(512, 128), BatchNorm1d(128), ReLU(),
            Linear(128, 5), BatchNorm1d(5), ReLU()
        )

    def forward(self, fmri, loading):
        fmri = self.fmri_pre_seq(fmri)
        fmri = torch.cat([self.fmri_feature0(fmri), self.fmri_feature1(fmri), self.fmri_feature2(fmri), self.fmri_feature3(fmri)], dim=-1)
        fmri = self.fmri_post_seq(fmri)

        loading = self.loading_seq(loading)

        x = torch.cat([fmri, loading], dim=-1)
        x = self.final_seq(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.__class__.__name__)

    def load(self, device=None):
        if device:
            self.load_state_dict(torch.load(self.__class__.__name__, map_location=device))
        else:
            self.load_state_dict(torch.load(self.__class__.__name__))
        self.eval()

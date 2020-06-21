import torch
from torch.nn import Module, Sequential, ReLU, Linear, Tanh, Dropout, BatchNorm1d, MaxPool3d, Conv3d, LeakyReLU, BatchNorm3d
from torch.nn.modules.flatten import Flatten


class Model(Module):
    def __init__(self):
        super().__init__()
        self.fmri_pre_seq = Sequential(
            Conv3d(53, 64, kernel_size=3), LeakyReLU(),
            Conv3d(64, 64, kernel_size=3), LeakyReLU(),
            Conv3d(64, 64, kernel_size=3), LeakyReLU(), BatchNorm3d(64),
            MaxPool3d(kernel_size=2, stride=2),
            Conv3d(64, 128, kernel_size=3), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), BatchNorm3d(128),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), Dropout(),
        )
        self.fmri_feature0 = Sequential(
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), Dropout(),
            Conv3d(128, 128, kernel_size=3, stride=1), BatchNorm3d(128), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            MaxPool3d(kernel_size=2, stride=2),
            Flatten(), Linear(14336, 256), ReLU()
        )
        self.fmri_feature1 = Sequential(
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), Dropout(),
            Conv3d(128, 128, kernel_size=3, stride=2), BatchNorm3d(128), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            MaxPool3d(kernel_size=2, stride=2),
            Flatten(), Linear(1536, 256), ReLU()
        )
        self.fmri_feature2 = Sequential(
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), Dropout(),
            Conv3d(128, 128, kernel_size=3, stride=4), BatchNorm3d(128), LeakyReLU(),
            Flatten(), Linear(4608, 256), ReLU()
        )
        self.fmri_feature3 = Sequential(
            Conv3d(128, 128, kernel_size=3), LeakyReLU(),
            Conv3d(128, 128, kernel_size=3), LeakyReLU(), Dropout(),
            Conv3d(128, 128, kernel_size=3, stride=8), BatchNorm3d(128), LeakyReLU(),
            Flatten(), Linear(1024, 256), ReLU()
        )
        self.fmri_post_seq = Sequential(
            Linear(1024, 1024), LeakyReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), LeakyReLU(),
            Linear(512, 512), BatchNorm1d(512)
        )
        self.loading_seq = Sequential(
            Linear(26, 256), LeakyReLU(),
            Linear(256, 1024), LeakyReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), LeakyReLU(),
            Linear(512, 512), LeakyReLU(), Dropout(),
            Linear(512, 512), LeakyReLU(), Dropout(),
            Linear(512, 512), BatchNorm1d(512), ReLU()
        )
        self.final_seq = Sequential(
            Linear(1024, 1024), LeakyReLU(), Dropout(),
            Linear(1024, 512), BatchNorm1d(512), LeakyReLU(),
            Linear(512, 512), BatchNorm1d(512), LeakyReLU(),
            Linear(512, 128), LeakyReLU(),
            Linear(128, 5), ReLU()
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
        self.load_state_dict(torch.load(self.__class__.__name__, map_location=device))
        self.to(device)
        self.eval()

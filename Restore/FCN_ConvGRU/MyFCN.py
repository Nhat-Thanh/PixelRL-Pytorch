import torch
import torch.nn as nn
import torch.nn.functional as F


class MyFCN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=4, dilation=4)

        # pi network
        self.conv_5_pi = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_pi = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2)
        self.W_xr = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.W_hr = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.W_xz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.W_hz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.W_xh = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.W_hh = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv_7_pi = nn.Conv2d(in_channels=64, out_channels=n_actions, kernel_size=3, padding=1)

        # v network
        self.conv_5_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2)
        self.conv_7_v = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def forward(self, X_in):
        X = X_in[:, 0:1, :, :]
        X = F.relu(self.conv_1(X))
        X = F.relu(self.conv_2(X))
        X = F.relu(self.conv_3(X))
        X = F.relu(self.conv_4(X))

        # pi network
        X_t = F.relu(self.conv_5_pi(X))
        X_t = F.relu(self.conv_6_pi(X_t))

        # ConvGRU
        H_t1 = X_in[:, -64:, :, :]
        R_t = torch.sigmoid(self.W_xr(X_t) + self.W_hr(H_t1))
        Z_t = torch.sigmoid(self.W_xz(X_t) + self.W_hz(H_t1))
        H_tilde_t = torch.tanh(self.W_xh(X_t) + self.W_hh(R_t * H_t1))
        H_t = Z_t * H_t1 + (1 - Z_t) * H_tilde_t

        pi = self.conv_7_pi(H_t)

        # v network
        X_v = F.relu(self.conv_5_v(X))
        X_v = F.relu(self.conv_6_v(X_v))
        v = self.conv_7_v(X_v)

        return pi, v, H_t

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
        self.conv_7_pi = nn.Conv2d(in_channels=64, out_channels=n_actions, kernel_size=3, padding=1)

        # v network
        self.conv_5_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_v = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2)
        self.conv_7_v = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

    def pi_and_v(self, X_in):
        X = F.relu(self.conv_1(X_in))
        X = F.relu(self.conv_2(X))
        X = F.relu(self.conv_3(X))
        X = F.relu(self.conv_4(X))

        # pi network
        X_pi = F.relu(self.conv_5_pi(X))
        X_pi = F.relu(self.conv_6_pi(X_pi))
        pi = torch.softmax(self.conv_7_pi(X_pi), dim=1)

        # v network
        X_v = F.relu(self.conv_5_v(X))
        X_v = F.relu(self.conv_6_v(X_v))
        v = self.conv_7_v(X_v)

        return pi, v 

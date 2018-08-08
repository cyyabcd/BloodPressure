import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseBlockBottleNeck(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BaseBlockBottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.input_channels, kernel_size = 1, padding = 0),
            nn.BatchNorm1d(self.input_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.input_channels, self.input_channels//2, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(self.input_channels//2),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.input_channels//2, self.output_channels, kernel_size = 1, padding = 0),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.output_channels, kernel_size = 5, padding = 2),
            )
        self.bn = nn.BatchNorm1d(self.output_channels)
    def forward(self, x):
        return self.bn(F.relu(self.conv1(x)+self.conv2(x), inplace = True))
import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(BaseBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.input_channels, kernel_size = 11, padding = 5),
            nn.BatchNorm1d(self.input_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.input_channels, self.input_channels, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(self.input_channels),
            nn.ReLU(inplace = True),
            nn.Conv1d(self.input_channels, self.output_channels, kernel_size = 5, padding = 2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.output_channels, kernel_size = 11, padding = 5),
            )
        self.bn = nn.BatchNorm1d(self.output_channels)
    def forward(self, x):
        return self.bn(F.relu(self.conv1(x)+self.conv2(x), inplace = True))
            
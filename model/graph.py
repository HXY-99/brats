import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, num_feature, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_feature, num_feature, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h.permute(0, 2, 1)).permute(0, 2, 1))
        return h


class BilateralGCN(nn.Module):
    def __init__(self, in_feature, num_node):
        super().__init__()
        self.gcn = GCN(in_feature, num_node)

    def forward(self, x, y):
        fusion = x + y
        fusion = self.gcn(fusion)
        x = x + fusion
        y = y + fusion

        return x, y


class TestModule(nn.Module):
    def __init__(self, in_channel):
        super(TestModule, self).__init__()
        self.num_node = in_channel // 4
        self.num_feature = in_channel // 2
        self.BGCN = BilateralGCN(self.num_feature, self.num_node)
        self.conv_v = nn.Conv2d(in_channel, self.num_node, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channel, self.num_feature, kernel_size=1)
        self.conv = nn.Conv2d(self.num_feature, in_channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x, y, x_fusion, y_fusion, concat=False):
        if concat:
            x = x + x_fusion
            y = y + y_fusion

        n, c, h, w = x.size()
        L = h * w
        x_v = self.conv_v(x).view(-1, self.num_node, L)
        y_v = self.conv_v(y).view(-1, self.num_node, L)
        x_w = self.conv_w(x).view(-1, self.num_feature, L)
        x_w = torch.transpose(x_w, 1, 2)
        y_w = self.conv_w(y).view(-1, self.num_feature, L)
        y_w = torch.transpose(y_w, 1, 2)
        x_graph = torch.bmm(x_v, x_w)
        y_graph = torch.bmm(y_v, y_w)
        x_graph = F.softmax(x_graph, dim=-1)
        y_graph = F.softmax(y_graph, dim=-1)
        x_graph, y_graph = self.BGCN(x_graph, y_graph)
        x_graph = torch.bmm(torch.transpose(x_v, 1, 2), x_graph).view(-1, self.num_feature, h, w)
        y_graph = torch.bmm(torch.transpose(y_v, 1, 2), y_graph).view(-1, self.num_feature, h, w)
        x_fusion = self.conv(x_graph)
        y_fusion = self.conv(y_graph)
        x_fusion = F.relu(self.bn(x_fusion))
        y_fusion = F.relu(self.bn(y_fusion))

        return x + x_fusion, y + y_fusion

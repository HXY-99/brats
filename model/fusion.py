from graph import *


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.in_channels != 1024:
            x = F.max_pool2d(x, 2, stride=2)
        y = F.relu(self.bn1(self.conv1(y)))
        if self.in_channels != 1024:
            y = F.max_pool2d(y, 2, stride=2)

        return x, y


class fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fusion, self).__init__()

        self.in_channels = in_channels
        self.reason = TestModule(in_channels)
        self.down = Downsample_block(in_channels, out_channels)

    def forward(self, seg1, edge1, seg2, edge2):
        if self.in_channels == 128:
            Seg, Edge = self.reason(seg1, edge1, seg2, edge2, concat=False)
        else:
            Seg, Edge = self.reason(seg1, edge1, seg2, edge2, concat=True)

        Seg, Edge = self.down(Seg, Edge)

        return Seg, Edge

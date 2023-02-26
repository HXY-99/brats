import torch.nn
from swintransformer import *
from fusion import *
from edge import Edgenet, ESAM


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.bn(self.conv(x)))

        return x


class whole_model(nn.Module):
    def __init__(self):
        super(whole_model, self).__init__()

        self.cnn = Edgenet()
        self.transformer = SwinTransformer()
        self.fusion1 = fusion(128, 256)
        self.fusion2 = fusion(256, 512)
        self.fusion3 = fusion(512, 1024)
        self.fusion4 = fusion(1024, 1024)
        self.ups1 = Upsample_block(1024, 512)
        self.ups2 = Upsample_block(512, 256)
        self.ups3 = Upsample_block(256, 128)
        self.ups4 = Upsample_block(128, 3)
        self.upe1 = ESAM(1024)
        self.upe2 = ESAM(512)
        self.upe3 = ESAM(256)
        self.upe4 = ESAM(128)


    def forward(self, in_seg, in_edge):

        # backbone
        seg_1, seg_2, seg_3, seg_4 = self.transformer(in_seg)  # 128, 256, 512, 1024
        edge_1, edge_2, edge_3, edge_4 = self.cnn(in_edge, feature=True)  # 128, 256, 512, 1024

        # fusion
        Seg_1, Edge_1 = self.fusion1(seg_1, edge_1, seg_1, edge_1)  # 128 ==> 256
        Seg_2, Edge_2 = self.fusion2(Seg_1, Edge_1, seg_2, edge_2)  # 256 ==> 512
        Seg_3, Edge_3 = self.fusion3(Seg_2, Edge_2, seg_3, edge_3)  # 512 ==> 1024
        Seg_4, Edge_4 = self.fusion4(Seg_3, Edge_3, seg_4, edge_4)  # 1024,1024 ==> 1024

        Seg = F.dropout2d(Seg_4)
        Seg = self.ups1(Seg, seg_3)     # 1024==>512
        Seg = self.ups2(Seg, seg_2)     # 512==>256
        Seg = self.ups3(Seg, seg_1)     # 256==>128
        Seg = self.ups4(Seg, in_seg)

        Edge_4 = self.upe1(Edge_4)
        Edge_3 = self.upe1(Edge_3)
        Edge_2 = self.upe2(Edge_2)
        Edge_1 = self.upe3(Edge_1)
        edge_1 = self.upe4(edge_1)

        Edge_4 = F.interpolate(Edge_4, scale_factor=16, mode="bilinear", align_corners=False)
        Edge_3 = F.interpolate(Edge_3, scale_factor=16, mode="bilinear", align_corners=False)
        Edge_2 = F.interpolate(Edge_2, scale_factor=8, mode="bilinear", align_corners=False)
        Edge_1 = F.interpolate(Edge_1, scale_factor=4, mode="bilinear", align_corners=False)
        edge_1 = F.interpolate(edge_1, scale_factor=2, mode="bilinear", align_corners=False)
        Edge = Edge_4 + Edge_3 + Edge_2 + Edge_1 + edge_1

        return Seg, Edge



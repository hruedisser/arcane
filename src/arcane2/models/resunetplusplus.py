import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ResUNet++ model implementation as in Rüdisser et al. 2022.
"""


class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SqueezeExciteBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        se = self.global_avg_pool(x)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn_shortcut = nn.BatchNorm1d(out_channels)
        self.squeeze_excite = SqueezeExciteBlock(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        shortcut = self.bn_shortcut(shortcut)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        x = x + shortcut
        x = self.squeeze_excite(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn_shortcut = nn.BatchNorm1d(out_channels)
        self.squeeze_excite = SqueezeExciteBlock(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        shortcut = self.bn_shortcut(shortcut)

        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))

        x = x + shortcut
        x = self.squeeze_excite(x)
        return x


class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate_scale=1):
        super(ASPPBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=6 * rate_scale,
            dilation=6 * rate_scale,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=12 * rate_scale,
            dilation=12 * rate_scale,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=18 * rate_scale,
            dilation=18 * rate_scale,
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.conv_out = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))
        x3 = self.bn3(self.conv3(x))
        x4 = self.bn4(self.conv4(x))

        y = x1 + x2 + x3 + x4
        y = self.conv_out(y)
        return y


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g_bn = nn.BatchNorm1d(in_channels)
        self.g_conv = nn.Conv1d(in_channels, in_channels * 4, kernel_size=3, padding=1)
        self.x_bn = nn.BatchNorm1d(in_channels * 4)
        self.x_conv = nn.Conv1d(
            in_channels * 4, in_channels * 4, kernel_size=3, padding=1
        )
        self.gc_bn = nn.BatchNorm1d(in_channels * 4)
        self.gc_conv = nn.Conv1d(
            in_channels * 4, in_channels * 4, kernel_size=3, padding=1
        )

    def forward(self, g, x):
        g_conv = self.g_conv(F.relu(self.g_bn(g)))
        g_pool = F.max_pool1d(g_conv, kernel_size=2, stride=4)

        x_conv = self.x_conv(F.relu(self.x_bn(x)))

        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(F.relu(self.gc_bn(gc_sum)))

        gc_mul = gc_conv * x
        return gc_mul


class ResUNet(nn.Module):
    def __init__(
        self,
        input_channels=8,
        input_length=1024,
        n_filters=[16, 32, 64, 128, 256, 512],
        num_classes=2,
        output_length=1024,
        activation="sigmoid",
    ):
        """
        ResUNet model implementation as in Rüdisser et al. 2022.
        Args:
            input_shape (tuple): Input shape of the model.
            n_filters (list): List of filter sizes for each block.
            num_classes (int): Number of classes for the output
        """

        super(ResUNet, self).__init__()

        input_shape = (input_channels, input_length)

        self.input_length = input_length
        self.output_length = output_length

        self.stem_block = StemBlock(input_shape[0], n_filters[2], stride=1)
        self.encoder_block1 = ResNetBlock(n_filters[2], n_filters[3], stride=4)
        self.encoder_block2 = ResNetBlock(n_filters[3], n_filters[4], stride=4)
        self.bridge = ASPPBlock(n_filters[4], n_filters[5])
        self.decoder_block1 = AttentionBlock(n_filters[3])
        self.decoder_resn1 = ResNetBlock(n_filters[5] + n_filters[3], n_filters[4])
        self.decoder_block2 = AttentionBlock(n_filters[2])
        self.decoder_resn2 = ResNetBlock(n_filters[4] + n_filters[2], n_filters[3])
        self.aspp_out = ASPPBlock(n_filters[3], n_filters[2])
        self.conv_out = nn.Conv1d(n_filters[2], num_classes, kernel_size=1)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "softmax":
            self.act = nn.Softmax(dim=1)

        if self.output_length < self.input_length:
            self.fc_out = nn.Linear(input_shape[1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)

        c1 = self.stem_block(x)
        c2 = self.encoder_block1(c1)
        c3 = self.encoder_block2(c2)
        b1 = self.bridge(c3)

        d1 = self.decoder_block1(c2, b1)
        d1 = F.interpolate(d1, scale_factor=4, mode="linear", align_corners=True)
        d1 = torch.cat([d1, c2], dim=1)
        d1 = self.decoder_resn1(d1)

        d2 = self.decoder_block2(c1, d1)
        d2 = F.interpolate(d2, scale_factor=4, mode="linear", align_corners=True)
        d2 = torch.cat([d2, c1], dim=1)
        d2 = self.decoder_resn2(d2)

        outputs = self.aspp_out(d2)
        outputs = self.conv_out(outputs)

        if self.output_length < self.input_length:
            outputs = self.fc_out(outputs)
        outputs = outputs.squeeze()
        outputs = self.act(outputs)

        return outputs

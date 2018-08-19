import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet-Based Translator

class Residual_Block(nn.Module):
    def __init__(self, channel, kernel_size=3, stride=1, padding=1, se_enable=True):
        super(Residual_Block, self).__init__()
        self.se_enable = se_enable

        self.conv1 = nn.Conv3d(channel, channel, kernel_size, stride, padding)
        self.conv1_norm = nn.InstanceNorm3d(channel)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size, stride, padding)
        self.conv2_norm = nn.InstanceNorm3d(channel)

        if se_enable:
            self.se_conv1 = nn.Conv3d(channel, channel // 16, kernel_size = 1)
            self.se_conv2 = nn.Conv3d(channel // 16, channel, kernel_size = 1)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = F.relu(self.conv2_norm(self.conv2(output)))

        if self.se_enable:
            se = F.avg_pool3d(output, (output.size(2), output.size(3), output.size(4)))
            se = F.relu(self.se_conv1(se))
            se = F.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        downsample = [nn.Conv3d(3, config.channel, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3)),
                      nn.InstanceNorm3d(config.channel, affine=True),
                      nn.ReLU(True)]
        for i in range(config.num_downsample):
            mult = 2**i
            downsample += [nn.Conv3d(config.channel * mult, config.channel * mult * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
                           nn.InstanceNorm3d(config.channel * mult * 2, affine=True),
                           nn.ReLU(True)]
        self.downsample = nn.Sequential(*downsample)

        resnet_blocks = []
        for i in range(config.num_block):
            resnet_blocks += [Residual_Block(config.channel*(2**config.num_downsample),
                                            kernel_size=3, stride=1, padding=1)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        upsample = []
        for i in range(config.num_downsample + 1):
            mult = 2**(config.num_downsample - i)
            upsample += [nn.ConvTranspose3d(config.channel * mult, config.channel * mult // 2, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0,1,1)),
                         nn.InstanceNorm3d(config.channel * mult // 2, affine=True),
                         nn.ReLU(True)]
        self.upsample = nn.Sequential(*upsample)
        self.final_conv = nn.Conv3d(config.channel//2, 3, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3))

    def forward(self, x):
        out = self.downsample(x)
        out = self.resnet_blocks(self.pool(out))
        out = self.upsample(out)
        out = self.final_conv(out)
        return out

# ResNet-based Translator

class ResNet_v2(nn.Module):
    def __init__(self, config):
        super(ResNet_v2, self).__init__()
        downsample = [nn.Conv3d(3, config.channel, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3)),
                      nn.InstanceNorm3d(config.channel, affine=True),
                      nn.ReLU(True)]
        for i in range(config.num_downsample):
            mult = 2**i
            downsample += [nn.Conv3d(config.channel * mult, config.channel * mult * 2, kernel_size=3, stride=1, padding=1),
                           nn.InstanceNorm3d(config.channel * mult * 2, affine=True),
                           nn.ReLU(True),
                           nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
        self.downsample = nn.Sequential(*downsample)

        resnet_blocks = []
        for i in range(config.num_block):
            resnet_blocks += [Residual_Block(config.channel*(2**config.num_downsample),
                                            kernel_size=3, stride=1, padding=1)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        upsample = []
        for i in range(config.num_downsample):
            mult = 2**(config.num_downsample - i)
            upsample += [nn.ConvTranspose3d(config.channel * mult, config.channel * mult // 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
                         nn.InstanceNorm3d(config.channel * mult // 2, affine=True),
                         nn.ReLU(True)]
        self.upsample = nn.Sequential(*upsample)
        self.final_conv = nn.Conv3d(config.channel, 3, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3))

    def forward(self, x):
        out = self.downsample(x)
        out = self.resnet_blocks(out)
        out = self.upsample(out)
        out = self.final_conv(out)
        return out

class ResNet_Basic(nn.Module):
    def __init__(self, config):
        super(ResNet_Basic, self).__init__()
        downsample = [nn.Conv3d(3, config.channel, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3)),
                      nn.InstanceNorm3d(config.channel, affine=True),
                      nn.ReLU(True)]
        for i in range(config.num_downsample):
            mult = 2**i
            downsample += [nn.Conv3d(config.channel * mult, config.channel * mult * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
                           nn.InstanceNorm3d(config.channel * mult * 2, affine=True),
                           nn.ReLU(True)]
        self.downsample = nn.Sequential(*downsample)

        resnet_blocks = []
        for i in range(config.num_block):
            resnet_blocks += [Residual_Block(config.channel*(2**config.num_downsample),
                                            kernel_size=3, stride=1, padding=1)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

        upsample = []
        for i in range(config.num_downsample):
            mult = 2**(config.num_downsample - i)
            upsample += [nn.ConvTranspose3d(config.channel * mult, config.channel * mult // 2, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0,1,1)),
                         nn.InstanceNorm3d(config.channel * mult // 2, affine=True),
                         nn.ReLU(True)]
        self.upsample = nn.Sequential(*upsample)
        self.final_conv = nn.Conv3d(config.channel, 3, kernel_size=(3, 7, 7), stride=1, padding=(1, 3, 3))

    def forward(self, x):
        out = self.downsample(x)
        out = self.resnet_blocks(out)
        out = self.upsample(out)
        out = self.final_conv(out)
        return out

# U-Net-Based Translator
# Based on https://arxiv.org/abs/1505.04597

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        no_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, no_pool

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_block=0):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = nn.ConvTranspose3d(self.in_channels, self.out_channels, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.conv1 = nn.Conv3d(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        resnet_blocks = []
        for i in range(num_block):
            resnet_blocks += [Residual_Block(self.out_channels, kernel_size=3, stride=1, padding=1)]
        self.resnet_blocks = nn.Sequential(*resnet_blocks)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        from_down = self.resnet_blocks(from_down)
        out = torch.cat((from_up, from_down), dim=1)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        self.channel = config.u_channel
        self.depth = config.u_depth

        self.down_convs = []
        self.up_convs = []

        for i in range(self.depth):
            ins = 3 if i == 0 else outs
            outs = self.channel * (2**i)
            pooling = True if i < self.depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(self.depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.final_conv = nn.Conv3d(self.channel, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        return self.final_conv(x)

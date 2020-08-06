###################################################################################################
#FPENet:Feature Pyramid Encoding Network for Real-time Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1909.08599v1.pdf
###################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from fvcore.nn.flop_count import flop_count #https://github.com/facebookresearch/fvcore
from tools.flops_counter.ptflops import get_model_complexity_info
from thop import profile #https://github.com/Lyken17/pytorch-OpCounter

__all__ = ["FPENet"]



def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups,bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class SEModule(nn.Module):
    # channel attention
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class FPEBlock(nn.Module):
    # feature pyramid encoding
    # t：膨胀系数; scales: 分组数量
    def __init__(self, inplanes, outplanes, dilat, downsample=None, stride=1, t=1, scales=4, se=False, norm_layer=None):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales,
                                            groups=(bottleneck_planes // scales),dilation=dilat[i],
                                            padding=1*dilat[i]) for i in range(scales)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales)])
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)   #divide
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s])))) #xs[0] >> conv3x3 >> bn >> relu >> ys[0]
            else:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1]))))#xs[1]+ys[0] >> conv3x3 >> bn >> relu >> ys[1]
        out = torch.cat(ys, 1)                  #concatenate

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out



class MEUModule(nn.Module):
    #mutual embedding upsample
    def __init__(self, channels_high, channels_low, channel_out):
        super(MEUModule, self).__init__()

        self.conv1x1_low = nn.Conv2d(channels_low, channel_out, kernel_size=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        self.conv1x1_high = nn.Conv2d(channels_high, channel_out, kernel_size=1, bias=False)
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(channel_out, channel_out, kernel_size=1, bias=False)

        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape

        #
        fms_low = self.conv1x1_low(fms_low)
        fms_low= self.bn_low(fms_low)
        #spatial attention
        sa_avg_out = self.sa_sigmoid(self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True)))
        #fms_low >> 1 channel >> conv1x1 >> Relu
        #
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        #channel attention
        ca_avg_out = self.ca_sigmoid(self.relu(self.ca_conv(self.avg_pool(fms_high))))
        # fms_high >> global average pooling >> conv1x1 >> Relu
        #
        fms_high_up = F.interpolate(fms_high, size=(h,w), mode='bilinear', align_corners=True)
        #multiplied by the upsampled high - level features
        fms_sa_att = sa_avg_out * fms_high_up
        #multiplied by the low - level features
        fms_ca_att = ca_avg_out * fms_low

        out = fms_ca_att + fms_sa_att

        return out


class FPENet(nn.Module):
    def __init__(self, classes=19, zero_init_residual=False,
                 width=16, scales=4, se=False, norm_layer=None):
        super(FPENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [int(width * 2 ** i) for i in range(3)] # planes=[16,32,64]

        self.block_num = [1,3,9]
        self.dilation = [1,2,4,8]

        self.inplanes = outplanes[0]
        #stage1
        self.conv1 = nn.Conv2d(3, outplanes[0], kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(FPEBlock, outplanes[0], self.block_num[0], dilation=self.dilation,
                                       stride=1, t=1, scales=scales, se=se, norm_layer=norm_layer)
        # stage2
        self.layer2 = self._make_layer(FPEBlock, outplanes[1], self.block_num[1], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)
        # stage3
        self.layer3 = self._make_layer(FPEBlock, outplanes[2], self.block_num[2], dilation=self.dilation,
                                       stride=2, t=4, scales=scales, se=se, norm_layer=norm_layer)
        self.meu1 = MEUModule(64,32,64)
        self.meu2 = MEUModule(64,16,32)

        # Projection layer
        self.project_layer = nn.Conv2d(32, classes, kernel_size = 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPEBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, dilation, stride=1, t=1, scales=4, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, dilat=dilation, downsample=downsample, stride=stride, t=t, scales=scales, se=se,
                            norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilat=dilation, scales=scales, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)

        ## stage 2
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2

        ## stage 3
        x_3_0 = self.layer3[0](x_2)
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8



        x2 = self.meu1(x_3, x_2)

        x1 = self.meu2(x2, x_1)

        output = self.project_layer(x1)

        # Bilinear interpolation x2
        output = F.interpolate(output,scale_factor=2, mode = 'bilinear', align_corners=True)

        return output


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPENet(classes=19).to(device)
    summary(model,(3,512,1024))
    x = torch.randn(1, 3, 512, 1024)

    from fvcore.nn.jit_handles import batchnorm_flop_jit
    from fvcore.nn.jit_handles import generic_activation_jit
    supported_ops = {
                        "aten::batch_norm": batchnorm_flop_jit,
                    }
    flop_dict, _ = flop_count(model, (x,), supported_ops)


    flops_count, params_count = get_model_complexity_info(model, (3, 512, 1024),
                                              as_strings=False,
                                              print_per_layer_stat=True)
    input = x
    macs, params = profile(model, inputs=(input,))
    print(flop_dict)
    print(flops_count, params_count)
    print(macs, params)
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 256, 512]             432
       BatchNorm2d-2         [-1, 16, 256, 512]              32
              ReLU-3         [-1, 16, 256, 512]               0              
            Conv2d-4         [-1, 16, 256, 512]             256
       BatchNorm2d-5         [-1, 16, 256, 512]              32
              ReLU-6         [-1, 16, 256, 512]               0              
            Conv2d-7          [-1, 4, 256, 512]              36
       BatchNorm2d-8          [-1, 4, 256, 512]               8
              ReLU-9          [-1, 4, 256, 512]               0
           Conv2d-10          [-1, 4, 256, 512]              36
      BatchNorm2d-11          [-1, 4, 256, 512]               8
             ReLU-12          [-1, 4, 256, 512]               0
           Conv2d-13          [-1, 4, 256, 512]              36
      BatchNorm2d-14          [-1, 4, 256, 512]               8
             ReLU-15          [-1, 4, 256, 512]               0
           Conv2d-16          [-1, 4, 256, 512]              36
      BatchNorm2d-17          [-1, 4, 256, 512]               8
             ReLU-18          [-1, 4, 256, 512]               0
           Conv2d-19         [-1, 16, 256, 512]             256
      BatchNorm2d-20         [-1, 16, 256, 512]              32
             ReLU-21         [-1, 16, 256, 512]               0
         FPEBlock-22         [-1, 16, 256, 512]               0
         
           Conv2d-23         [-1, 64, 128, 256]           1,024
      BatchNorm2d-24         [-1, 64, 128, 256]             128
             ReLU-25         [-1, 64, 128, 256]               0
           Conv2d-26         [-1, 16, 128, 256]             144
      BatchNorm2d-27         [-1, 16, 128, 256]              32
             ReLU-28         [-1, 16, 128, 256]               0
           Conv2d-29         [-1, 16, 128, 256]             144
      BatchNorm2d-30         [-1, 16, 128, 256]              32
             ReLU-31         [-1, 16, 128, 256]               0
           Conv2d-32         [-1, 16, 128, 256]             144
      BatchNorm2d-33         [-1, 16, 128, 256]              32
             ReLU-34         [-1, 16, 128, 256]               0
           Conv2d-35         [-1, 16, 128, 256]             144
      BatchNorm2d-36         [-1, 16, 128, 256]              32
             ReLU-37         [-1, 16, 128, 256]               0
           Conv2d-38         [-1, 32, 128, 256]           2,048
      BatchNorm2d-39         [-1, 32, 128, 256]              64
           Conv2d-40         [-1, 32, 128, 256]             512
      BatchNorm2d-41         [-1, 32, 128, 256]              64
             ReLU-42         [-1, 32, 128, 256]               0
         FPEBlock-43         [-1, 32, 128, 256]               0
           Conv2d-44         [-1, 32, 128, 256]           1,024
      BatchNorm2d-45         [-1, 32, 128, 256]              64
             ReLU-46         [-1, 32, 128, 256]               0
           Conv2d-47          [-1, 8, 128, 256]              72
      BatchNorm2d-48          [-1, 8, 128, 256]              16
             ReLU-49          [-1, 8, 128, 256]               0
           Conv2d-50          [-1, 8, 128, 256]              72
      BatchNorm2d-51          [-1, 8, 128, 256]              16
             ReLU-52          [-1, 8, 128, 256]               0
           Conv2d-53          [-1, 8, 128, 256]              72
      BatchNorm2d-54          [-1, 8, 128, 256]              16
             ReLU-55          [-1, 8, 128, 256]               0
           Conv2d-56          [-1, 8, 128, 256]              72
      BatchNorm2d-57          [-1, 8, 128, 256]              16
             ReLU-58          [-1, 8, 128, 256]               0
           Conv2d-59         [-1, 32, 128, 256]           1,024
      BatchNorm2d-60         [-1, 32, 128, 256]              64
             ReLU-61         [-1, 32, 128, 256]               0
         FPEBlock-62         [-1, 32, 128, 256]               0
           Conv2d-63         [-1, 32, 128, 256]           1,024
      BatchNorm2d-64         [-1, 32, 128, 256]              64
             ReLU-65         [-1, 32, 128, 256]               0
           Conv2d-66          [-1, 8, 128, 256]              72
      BatchNorm2d-67          [-1, 8, 128, 256]              16
             ReLU-68          [-1, 8, 128, 256]               0
           Conv2d-69          [-1, 8, 128, 256]              72
      BatchNorm2d-70          [-1, 8, 128, 256]              16
             ReLU-71          [-1, 8, 128, 256]               0
           Conv2d-72          [-1, 8, 128, 256]              72
      BatchNorm2d-73          [-1, 8, 128, 256]              16
             ReLU-74          [-1, 8, 128, 256]               0
           Conv2d-75          [-1, 8, 128, 256]              72
      BatchNorm2d-76          [-1, 8, 128, 256]              16
             ReLU-77          [-1, 8, 128, 256]               0
           Conv2d-78         [-1, 32, 128, 256]           1,024
      BatchNorm2d-79         [-1, 32, 128, 256]              64
             ReLU-80         [-1, 32, 128, 256]               0
         FPEBlock-81         [-1, 32, 128, 256]               0
         
           Conv2d-82         [-1, 128, 64, 128]           4,096
      BatchNorm2d-83         [-1, 128, 64, 128]             256
             ReLU-84         [-1, 128, 64, 128]               0
           Conv2d-85          [-1, 32, 64, 128]             288
      BatchNorm2d-86          [-1, 32, 64, 128]              64
             ReLU-87          [-1, 32, 64, 128]               0
           Conv2d-88          [-1, 32, 64, 128]             288
      BatchNorm2d-89          [-1, 32, 64, 128]              64
             ReLU-90          [-1, 32, 64, 128]               0
           Conv2d-91          [-1, 32, 64, 128]             288
      BatchNorm2d-92          [-1, 32, 64, 128]              64
             ReLU-93          [-1, 32, 64, 128]               0
           Conv2d-94          [-1, 32, 64, 128]             288
      BatchNorm2d-95          [-1, 32, 64, 128]              64
             ReLU-96          [-1, 32, 64, 128]               0
           Conv2d-97          [-1, 64, 64, 128]           8,192
      BatchNorm2d-98          [-1, 64, 64, 128]             128
           Conv2d-99          [-1, 64, 64, 128]           2,048
     BatchNorm2d-100          [-1, 64, 64, 128]             128
            ReLU-101          [-1, 64, 64, 128]               0
        FPEBlock-102          [-1, 64, 64, 128]               0
          Conv2d-103          [-1, 64, 64, 128]           4,096
     BatchNorm2d-104          [-1, 64, 64, 128]             128
            ReLU-105          [-1, 64, 64, 128]               0
          Conv2d-106          [-1, 16, 64, 128]             144
     BatchNorm2d-107          [-1, 16, 64, 128]              32
            ReLU-108          [-1, 16, 64, 128]               0
          Conv2d-109          [-1, 16, 64, 128]             144
     BatchNorm2d-110          [-1, 16, 64, 128]              32
            ReLU-111          [-1, 16, 64, 128]               0
          Conv2d-112          [-1, 16, 64, 128]             144
     BatchNorm2d-113          [-1, 16, 64, 128]              32
            ReLU-114          [-1, 16, 64, 128]               0
          Conv2d-115          [-1, 16, 64, 128]             144
     BatchNorm2d-116          [-1, 16, 64, 128]              32
            ReLU-117          [-1, 16, 64, 128]               0
          Conv2d-118          [-1, 64, 64, 128]           4,096
     BatchNorm2d-119          [-1, 64, 64, 128]             128
            ReLU-120          [-1, 64, 64, 128]               0
        FPEBlock-121          [-1, 64, 64, 128]               0
          Conv2d-122          [-1, 64, 64, 128]           4,096
     BatchNorm2d-123          [-1, 64, 64, 128]             128
            ReLU-124          [-1, 64, 64, 128]               0
          Conv2d-125          [-1, 16, 64, 128]             144
     BatchNorm2d-126          [-1, 16, 64, 128]              32
            ReLU-127          [-1, 16, 64, 128]               0
          Conv2d-128          [-1, 16, 64, 128]             144
     BatchNorm2d-129          [-1, 16, 64, 128]              32
            ReLU-130          [-1, 16, 64, 128]               0
          Conv2d-131          [-1, 16, 64, 128]             144
     BatchNorm2d-132          [-1, 16, 64, 128]              32
            ReLU-133          [-1, 16, 64, 128]               0
          Conv2d-134          [-1, 16, 64, 128]             144
     BatchNorm2d-135          [-1, 16, 64, 128]              32
            ReLU-136          [-1, 16, 64, 128]               0
          Conv2d-137          [-1, 64, 64, 128]           4,096
     BatchNorm2d-138          [-1, 64, 64, 128]             128
            ReLU-139          [-1, 64, 64, 128]               0
        FPEBlock-140          [-1, 64, 64, 128]               0
          Conv2d-141          [-1, 64, 64, 128]           4,096
     BatchNorm2d-142          [-1, 64, 64, 128]             128
            ReLU-143          [-1, 64, 64, 128]               0
          Conv2d-144          [-1, 16, 64, 128]             144
     BatchNorm2d-145          [-1, 16, 64, 128]              32
            ReLU-146          [-1, 16, 64, 128]               0
          Conv2d-147          [-1, 16, 64, 128]             144
     BatchNorm2d-148          [-1, 16, 64, 128]              32
            ReLU-149          [-1, 16, 64, 128]               0
          Conv2d-150          [-1, 16, 64, 128]             144
     BatchNorm2d-151          [-1, 16, 64, 128]              32
            ReLU-152          [-1, 16, 64, 128]               0
          Conv2d-153          [-1, 16, 64, 128]             144
     BatchNorm2d-154          [-1, 16, 64, 128]              32
            ReLU-155          [-1, 16, 64, 128]               0
          Conv2d-156          [-1, 64, 64, 128]           4,096
     BatchNorm2d-157          [-1, 64, 64, 128]             128
            ReLU-158          [-1, 64, 64, 128]               0
        FPEBlock-159          [-1, 64, 64, 128]               0
          Conv2d-160          [-1, 64, 64, 128]           4,096
     BatchNorm2d-161          [-1, 64, 64, 128]             128
            ReLU-162          [-1, 64, 64, 128]               0
          Conv2d-163          [-1, 16, 64, 128]             144
     BatchNorm2d-164          [-1, 16, 64, 128]              32
            ReLU-165          [-1, 16, 64, 128]               0
          Conv2d-166          [-1, 16, 64, 128]             144
     BatchNorm2d-167          [-1, 16, 64, 128]              32
            ReLU-168          [-1, 16, 64, 128]               0
          Conv2d-169          [-1, 16, 64, 128]             144
     BatchNorm2d-170          [-1, 16, 64, 128]              32
            ReLU-171          [-1, 16, 64, 128]               0
          Conv2d-172          [-1, 16, 64, 128]             144
     BatchNorm2d-173          [-1, 16, 64, 128]              32
            ReLU-174          [-1, 16, 64, 128]               0
          Conv2d-175          [-1, 64, 64, 128]           4,096
     BatchNorm2d-176          [-1, 64, 64, 128]             128
            ReLU-177          [-1, 64, 64, 128]               0
        FPEBlock-178          [-1, 64, 64, 128]               0
          Conv2d-179          [-1, 64, 64, 128]           4,096
     BatchNorm2d-180          [-1, 64, 64, 128]             128
            ReLU-181          [-1, 64, 64, 128]               0
          Conv2d-182          [-1, 16, 64, 128]             144
     BatchNorm2d-183          [-1, 16, 64, 128]              32
            ReLU-184          [-1, 16, 64, 128]               0
          Conv2d-185          [-1, 16, 64, 128]             144
     BatchNorm2d-186          [-1, 16, 64, 128]              32
            ReLU-187          [-1, 16, 64, 128]               0
          Conv2d-188          [-1, 16, 64, 128]             144
     BatchNorm2d-189          [-1, 16, 64, 128]              32
            ReLU-190          [-1, 16, 64, 128]               0
          Conv2d-191          [-1, 16, 64, 128]             144
     BatchNorm2d-192          [-1, 16, 64, 128]              32
            ReLU-193          [-1, 16, 64, 128]               0
          Conv2d-194          [-1, 64, 64, 128]           4,096
     BatchNorm2d-195          [-1, 64, 64, 128]             128
            ReLU-196          [-1, 64, 64, 128]               0
        FPEBlock-197          [-1, 64, 64, 128]               0
          Conv2d-198          [-1, 64, 64, 128]           4,096
     BatchNorm2d-199          [-1, 64, 64, 128]             128
            ReLU-200          [-1, 64, 64, 128]               0
          Conv2d-201          [-1, 16, 64, 128]             144
     BatchNorm2d-202          [-1, 16, 64, 128]              32
            ReLU-203          [-1, 16, 64, 128]               0
          Conv2d-204          [-1, 16, 64, 128]             144
     BatchNorm2d-205          [-1, 16, 64, 128]              32
            ReLU-206          [-1, 16, 64, 128]               0
          Conv2d-207          [-1, 16, 64, 128]             144
     BatchNorm2d-208          [-1, 16, 64, 128]              32
            ReLU-209          [-1, 16, 64, 128]               0
          Conv2d-210          [-1, 16, 64, 128]             144
     BatchNorm2d-211          [-1, 16, 64, 128]              32
            ReLU-212          [-1, 16, 64, 128]               0
          Conv2d-213          [-1, 64, 64, 128]           4,096
     BatchNorm2d-214          [-1, 64, 64, 128]             128
            ReLU-215          [-1, 64, 64, 128]               0
        FPEBlock-216          [-1, 64, 64, 128]               0
          Conv2d-217          [-1, 64, 64, 128]           4,096
     BatchNorm2d-218          [-1, 64, 64, 128]             128
            ReLU-219          [-1, 64, 64, 128]               0
          Conv2d-220          [-1, 16, 64, 128]             144
     BatchNorm2d-221          [-1, 16, 64, 128]              32
            ReLU-222          [-1, 16, 64, 128]               0
          Conv2d-223          [-1, 16, 64, 128]             144
     BatchNorm2d-224          [-1, 16, 64, 128]              32
            ReLU-225          [-1, 16, 64, 128]               0
          Conv2d-226          [-1, 16, 64, 128]             144
     BatchNorm2d-227          [-1, 16, 64, 128]              32
            ReLU-228          [-1, 16, 64, 128]               0
          Conv2d-229          [-1, 16, 64, 128]             144
     BatchNorm2d-230          [-1, 16, 64, 128]              32
            ReLU-231          [-1, 16, 64, 128]               0
          Conv2d-232          [-1, 64, 64, 128]           4,096
     BatchNorm2d-233          [-1, 64, 64, 128]             128
            ReLU-234          [-1, 64, 64, 128]               0
        FPEBlock-235          [-1, 64, 64, 128]               0
          Conv2d-236          [-1, 64, 64, 128]           4,096
     BatchNorm2d-237          [-1, 64, 64, 128]             128
            ReLU-238          [-1, 64, 64, 128]               0
          Conv2d-239          [-1, 16, 64, 128]             144
     BatchNorm2d-240          [-1, 16, 64, 128]              32
            ReLU-241          [-1, 16, 64, 128]               0
          Conv2d-242          [-1, 16, 64, 128]             144
     BatchNorm2d-243          [-1, 16, 64, 128]              32
            ReLU-244          [-1, 16, 64, 128]               0
          Conv2d-245          [-1, 16, 64, 128]             144
     BatchNorm2d-246          [-1, 16, 64, 128]              32
            ReLU-247          [-1, 16, 64, 128]               0
          Conv2d-248          [-1, 16, 64, 128]             144
     BatchNorm2d-249          [-1, 16, 64, 128]              32
            ReLU-250          [-1, 16, 64, 128]               0
          Conv2d-251          [-1, 64, 64, 128]           4,096
     BatchNorm2d-252          [-1, 64, 64, 128]             128
            ReLU-253          [-1, 64, 64, 128]               0
        FPEBlock-254          [-1, 64, 64, 128]               0
        
          Conv2d-255         [-1, 64, 128, 256]           2,048
     BatchNorm2d-256         [-1, 64, 128, 256]             128
          Conv2d-257          [-1, 1, 128, 256]               1
         Sigmoid-258          [-1, 1, 128, 256]               0
          Conv2d-259          [-1, 64, 64, 128]           4,096
     BatchNorm2d-260          [-1, 64, 64, 128]             128
AdaptiveAvgPool2d-261             [-1, 64, 1, 1]               0
          Conv2d-262             [-1, 64, 1, 1]           4,096
            ReLU-263             [-1, 64, 1, 1]               0
         Sigmoid-264             [-1, 64, 1, 1]               0
       MEUModule-265         [-1, 64, 128, 256]               0
          Conv2d-266         [-1, 32, 256, 512]             512
     BatchNorm2d-267         [-1, 32, 256, 512]              64
          Conv2d-268          [-1, 1, 256, 512]               1
         Sigmoid-269          [-1, 1, 256, 512]               0
          Conv2d-270         [-1, 32, 128, 256]           2,048
     BatchNorm2d-271         [-1, 32, 128, 256]              64
AdaptiveAvgPool2d-272             [-1, 32, 1, 1]               0
          Conv2d-273             [-1, 32, 1, 1]           1,024
            ReLU-274             [-1, 32, 1, 1]               0
         Sigmoid-275             [-1, 32, 1, 1]               0
       MEUModule-276         [-1, 32, 256, 512]               0
          Conv2d-277         [-1, 19, 256, 512]             627
================================================================
Total params: 115,125
Trainable params: 115,125
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 6.00
Forward/backward pass size (MB): 1093.50
Params size (MB): 0.44
Estimated Total Size (MB): 1099.94
----------------------------------------------------------------
注，下面数据按11 classes 计算而得
FPENet(                     
  1.542 GMac, 100.000% MACs, 
  #第一阶段，原始图像/2得到16特征，1个FPE保持特征shape不变  
  (conv1): Conv2d(0.057 GMac, 3.672% MACs, 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(0.004 GMac, 0.272% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(0.002 GMac, 0.136% MACs, inplace=True)
  (layer1): Sequential(
    0.105 GMac, 6.800% MACs, 
    (0): FPEBlock(
      0.105 GMac, 6.800% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.004 GMac, 0.272% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.019 GMac, 1.224% MACs, 
        (0): Conv2d(0.005 GMac, 0.306% MACs, 4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)
        (1): Conv2d(0.005 GMac, 0.306% MACs, 4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=4, bias=False)
        (2): Conv2d(0.005 GMac, 0.306% MACs, 4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=4, bias=False)
        (3): Conv2d(0.005 GMac, 0.306% MACs, 4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=4, bias=False)
      )
      (bn2): ModuleList(
        0.004 GMac, 0.272% MACs, 
        (0): BatchNorm2d(0.001 GMac, 0.068% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.001 GMac, 0.068% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.001 GMac, 0.068% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.001 GMac, 0.068% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.004 GMac, 0.272% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.006 GMac, 0.408% MACs, inplace=True)
    )
  )
  #第二阶段，1个FPE特征图/2得到32特征，2个FPE保持特征shape不变 
  (layer2): Sequential(
    0.326 GMac, 21.149% MACs, 
    (0): FPEBlock(                  #downsample
      0.154 GMac, 9.996% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 16, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (bn1): BatchNorm2d(0.004 GMac, 0.272% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.019 GMac, 1.224% MACs, 
        (0): Conv2d(0.005 GMac, 0.306% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.005 GMac, 0.306% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.005 GMac, 0.306% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.005 GMac, 0.306% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.004 GMac, 0.272% MACs, 
        (0): BatchNorm2d(0.001 GMac, 0.068% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.001 GMac, 0.068% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.001 GMac, 0.068% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.001 GMac, 0.068% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.067 GMac, 4.352% MACs, 64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.005 GMac, 0.340% MACs, inplace=True)
      (downsample): Sequential(
        0.019 GMac, 1.224% MACs, 
        (0): Conv2d(0.017 GMac, 1.088% MACs, 16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): FPEBlock(
      0.086 GMac, 5.576% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.009 GMac, 0.612% MACs, 
        (0): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        (1): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=8, bias=False)
        (2): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=8, bias=False)
        (3): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=8, bias=False)
      )
      (bn2): ModuleList(
        0.002 GMac, 0.136% MACs, 
        (0): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.003 GMac, 0.204% MACs, inplace=True)
    )
    (2): FPEBlock(
      0.086 GMac, 5.576% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.009 GMac, 0.612% MACs, 
        (0): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        (1): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=8, bias=False)
        (2): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=8, bias=False)
        (3): Conv2d(0.002 GMac, 0.153% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=8, bias=False)
      )
      (bn2): ModuleList(
        0.002 GMac, 0.136% MACs, 
        (0): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.001 GMac, 0.034% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.003 GMac, 0.204% MACs, inplace=True)
    )
  )
  #第三阶段，1个FPE特征图/2得到64特征，8个FPE保持特征shape不变
  (layer3): Sequential(
    0.748 GMac, 48.520% MACs, 
    (0): FPEBlock(                      #downsample
      0.136 GMac, 8.806% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 32, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (bn1): BatchNorm2d(0.002 GMac, 0.136% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.009 GMac, 0.612% MACs, 
        (0): Conv2d(0.002 GMac, 0.153% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (1): Conv2d(0.002 GMac, 0.153% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=32, bias=False)
        (2): Conv2d(0.002 GMac, 0.153% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=32, bias=False)
        (3): Conv2d(0.002 GMac, 0.153% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=32, bias=False)
      )
      (bn2): ModuleList(
        0.002 GMac, 0.136% MACs, 
        (0): BatchNorm2d(0.001 GMac, 0.034% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.001 GMac, 0.034% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.001 GMac, 0.034% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.001 GMac, 0.034% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.067 GMac, 4.352% MACs, 128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.003 GMac, 0.170% MACs, inplace=True)
      (downsample): Sequential(
        0.018 GMac, 1.156% MACs, 
        (0): Conv2d(0.017 GMac, 1.088% MACs, 32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (2): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (3): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (4): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (5): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (6): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (7): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
    (8): FPEBlock(
      0.077 GMac, 4.964% MACs, 
      (conv1): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): ModuleList(
        0.005 GMac, 0.306% MACs, 
        (0): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
        (1): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=16, bias=False)
        (2): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), groups=16, bias=False)
        (3): Conv2d(0.001 GMac, 0.077% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8), groups=16, bias=False)
      )
      (bn2): ModuleList(
        0.001 GMac, 0.068% MACs, 
        (0): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): BatchNorm2d(0.0 GMac, 0.017% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.002 GMac, 0.102% MACs, inplace=True)
    )
  )
  #融合一阶段，由第二阶段和第三阶段的32特征和64特征融合
  #第二阶段低级32特征调整到64特征，分辨率不变
  #第二阶段低级64特征生成1特征空间注意力，分辨率不变
  
  #第三阶段高级64特征调整到64特征后，
  #1.分辨率*2
  #2.生成特征图=1的64通道注意力
  
  #权重交叉相乘后再相加
  (meu1): MEUModule(
    0.106 GMac, 6.905% MACs, 
    (conv1x1_low): Conv2d(0.067 GMac, 4.352% MACs, 32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn_low): BatchNorm2d(0.004 GMac, 0.272% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (sa_conv): Conv2d(0.0 GMac, 0.002% MACs, 1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (conv1x1_high): Conv2d(0.034 GMac, 2.176% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn_high): BatchNorm2d(0.001 GMac, 0.068% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (avg_pool): AdaptiveAvgPool2d(0.001 GMac, 0.034% MACs, output_size=1)
    (ca_conv): Conv2d(0.0 GMac, 0.000% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (sa_sigmoid): Sigmoid(0.0 GMac, 0.000% MACs, )
    (ca_sigmoid): Sigmoid(0.0 GMac, 0.000% MACs, )
    (relu): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
  )
  (meu2): MEUModule(
    0.146 GMac, 9.461% MACs, 
    (conv1x1_low): Conv2d(0.067 GMac, 4.352% MACs, 16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn_low): BatchNorm2d(0.008 GMac, 0.544% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (sa_conv): Conv2d(0.0 GMac, 0.009% MACs, 1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (conv1x1_high): Conv2d(0.067 GMac, 4.352% MACs, 64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn_high): BatchNorm2d(0.002 GMac, 0.136% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (avg_pool): AdaptiveAvgPool2d(0.001 GMac, 0.068% MACs, output_size=1)
    (ca_conv): Conv2d(0.0 GMac, 0.000% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (sa_sigmoid): Sigmoid(0.0 GMac, 0.000% MACs, )
    (ca_sigmoid): Sigmoid(0.0 GMac, 0.000% MACs, )
    (relu): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
  )
  (project_layer): Conv2d(0.048 GMac, 3.086% MACs, 32, 11, kernel_size=(1, 1), stride=(1, 1))
)
    #最后插值得到原始的分辨率
    
    train_time 2.95
    val_time 0.53
'''


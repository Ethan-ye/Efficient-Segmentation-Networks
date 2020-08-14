# *- coding: utf-8 -*

###########################################################################
# Partial order pruning: for best speed/accuracy trade-off in neural architecture search
# https://github.com/lixincn2015/Partial-Order-Pruning
###########################################################################
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from utils.activations import NON_LINEARITY
from fvcore.nn.flop_count import flop_count  # https://github.com/facebookresearch/fvcore
from tools.flops_counter.ptflops import get_model_complexity_info
from thop import profile  # https://github.com/Lyken17/pytorch-OpCounter

__all__ = ['DF1SegG']


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if stride == 2:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=6, stride=2, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = self.relu(out)

        return out

        # self.decoder4 = nn.Sequential(
        #     nn.Conv2d(128, 32, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, groups=32, bias=False)  # lr=0, bilinear
        # )


class FuseBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dconv = nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1, groups=planes,
                                        bias=False)  # lr=0, bilinear
        self.conv2 = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.stride = stride

    def forward(self, deep, shallow):
        deep = self.conv1(deep)
        deep = self.bn1(deep)
        deep = self.relu(deep)
        # with torch.no_grad():
        deep = self.dconv(deep)
        fuse = torch.cat([deep, shallow], 1)
        fuse = self.conv2(fuse)
        fuse = self.bn2(fuse)
        fuse = self.relu(fuse)

        return fuse


# https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py
class PSPModule(nn.Module):
    def __init__(self, features=512, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        n = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size, n) for size in sizes])
        self.bottleneck = nn.Conv2d(features + features // n * n, out_features, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size, n):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features // n, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(features // n)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        priors = [F.upsample(input=stage(x), size=(h, w), mode='bilinear') for stage in self.stages] + [x]
        out = self.bottleneck(torch.cat(priors, 1))
        out = self.bn(out)
        return self.relu(out)


class DF1SegG(nn.Module):
    def __init__(self, classes=19):
        super(DF1SegG, self).__init__()
        # encode
        # Downsample
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv2d(3, 32, kernel_size=6, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Downsample
        self.conv2 = nn.Sequential(
            # nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv2d(32, 64, kernel_size=6, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.inplanes = 64
        # Downsample
        self.res2 = self._make_layer(64, 3, stride=2)
        # Downsample
        self.res3 = self._make_layer(128, 3, stride=2)
        # Downsample
        self.res4_1 = self._make_layer(256, 3, stride=2)
        self.res4_2 = self._make_layer(512, 1, stride=1)

        self.psp = PSPModule(512, 512, (1, 2, 4, 8))

        self.wc3 = nn.Sequential(
            nn.Conv2d(64, classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(classes),
            nn.ReLU(inplace=True)
        )
        self.wc4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.wc5 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec4 = FuseBlock(128, 32)  # fuse wc5 wc4
        self.dec3 = FuseBlock(32, classes)  # fuse wc4 wc3

        self.score = nn.Sequential(
            nn.Conv2d(classes, classes, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(classes),
            nn.ReLU(inplace=True)
        )

        self.score_u8 = nn.ConvTranspose2d(classes, classes, kernel_size=16, padding=4, stride=8, groups=classes,
                                           bias=False)

        self._initialize_weights()
        # decode

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        if stride != 1 or self.inplanes != planes:#dowmsample or channel adjust
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            downsample = None

        layers = list()
        layers.append(ResBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     m.weight.data.zero_()
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels//m.groups, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_res2 = self.res2(x)
        x_res3 = self.res3(x_res2)
        x = self.res4_1(x_res3)
        x = self.res4_2(x)
        x = self.psp(x)
        x_wc3 = self.wc3(x_res2)
        x_wc4 = self.wc4(x_res3)
        x = self.wc5(x)
        x = self.dec4(x, x_wc4)
        x = self.dec3(x, x_wc3)
        x = self.score(x)
        # with torch.no_grad():
        x = self.score_u8(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DF1SegG(classes=19).to(device)
    summary(model, (3, 352, 480))
    x = torch.randn(2, 3, 512, 1024)

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
================================================================
Total params: 13,112,923
Trainable params: 13,112,923
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 6.00
Forward/backward pass size (MB): 427.41
Params size (MB): 50.02
Estimated Total Size (MB): 483.43
----------------------------------------------------------------

DF1Seg(
  10.476 GMac, 100.000% MACs, 
  (conv1): Sequential(
    0.126 GMac, 1.201% MACs, 
    (0): Conv2d(0.113 GMac, 1.081% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.008 GMac, 0.080% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.004 GMac, 0.040% MACs, inplace=True)
  )
  (conv2): Sequential(
    0.61 GMac, 5.826% MACs, 
    (0): Conv2d(0.604 GMac, 5.766% MACs, 32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.004 GMac, 0.040% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.002 GMac, 0.020% MACs, inplace=True)
  )
  (res2): Sequential(
    1.856 GMac, 17.717% MACs, 
    (0): ResBlock(
      0.642 GMac, 6.126% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.010% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        0.035 GMac, 0.330% MACs, 
        (0): Conv2d(0.034 GMac, 0.320% MACs, 64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResBlock(
      0.607 GMac, 5.796% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.010% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): ResBlock(
      0.607 GMac, 5.796% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.010% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.010% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (res3): Sequential(
    1.683 GMac, 16.066% MACs, 
    (0): ResBlock(
      0.472 GMac, 4.504% MACs, 
      (conv1): Conv2d(0.151 GMac, 1.441% MACs, 64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.005% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        0.017 GMac, 0.165% MACs, 
        (0): Conv2d(0.017 GMac, 0.160% MACs, 64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResBlock(
      0.606 GMac, 5.781% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.005% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): ResBlock(
      0.606 GMac, 5.781% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.005% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.005% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (res4_1): Sequential(
    1.68 GMac, 16.041% MACs, 
    (0): ResBlock(
      0.471 GMac, 4.494% MACs, 
      (conv1): Conv2d(0.151 GMac, 1.441% MACs, 128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.0 GMac, 0.003% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        0.017 GMac, 0.163% MACs, 
        (0): Conv2d(0.017 GMac, 0.160% MACs, 128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResBlock(
      0.605 GMac, 5.773% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.0 GMac, 0.003% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): ResBlock(
      0.605 GMac, 5.773% MACs, 
      (conv1): Conv2d(0.302 GMac, 2.883% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.0 GMac, 0.003% MACs, inplace=True)
      (conv2): Conv2d(0.302 GMac, 2.883% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (res4_2): Sequential(
    1.881 GMac, 17.957% MACs, 
    (0): ResBlock(
      1.881 GMac, 17.957% MACs, 
      (conv1): Conv2d(0.604 GMac, 5.766% MACs, 256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(0.001 GMac, 0.005% MACs, inplace=True)
      (conv2): Conv2d(1.208 GMac, 11.531% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(0.001 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        0.068 GMac, 0.646% MACs, 
        (0): Conv2d(0.067 GMac, 0.641% MACs, 256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (psp): PSPModule(
    2.423 GMac, 23.133% MACs, 
    (stages): ModuleList(
      0.007 GMac, 0.063% MACs, 
      (0): Sequential(
        0.0 GMac, 0.003% MACs, 
        (0): AdaptiveAvgPool2d(0.0 GMac, 0.003% MACs, output_size=(1, 1))
        (1): Conv2d(0.0 GMac, 0.001% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
      )
      (1): Sequential(
        0.001 GMac, 0.005% MACs, 
        (0): AdaptiveAvgPool2d(0.0 GMac, 0.003% MACs, output_size=(2, 2))
        (1): Conv2d(0.0 GMac, 0.003% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
      )
      (2): Sequential(
        0.001 GMac, 0.013% MACs, 
        (0): AdaptiveAvgPool2d(0.0 GMac, 0.003% MACs, output_size=(4, 4))
        (1): Conv2d(0.001 GMac, 0.010% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
      )
      (3): Sequential(
        0.004 GMac, 0.043% MACs, 
        (0): AdaptiveAvgPool2d(0.0 GMac, 0.003% MACs, output_size=(8, 8))
        (1): Conv2d(0.004 GMac, 0.040% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(0.0 GMac, 0.000% MACs, inplace=True)
      )
    )
    (bottleneck): Conv2d(2.416 GMac, 23.062% MACs, 1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(0.001 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(0.0 GMac, 0.003% MACs, inplace=True)
  )
  (wc3): Sequential(
    0.01 GMac, 0.100% MACs, 
    (0): Conv2d(0.01 GMac, 0.095% MACs, 64, 19, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 GMac, 0.003% MACs, 19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 GMac, 0.001% MACs, inplace=True)
  )
  (wc4): Sequential(
    0.009 GMac, 0.082% MACs, 
    (0): Conv2d(0.008 GMac, 0.080% MACs, 128, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 GMac, 0.001% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 GMac, 0.001% MACs, inplace=True)
  )
  (wc5): Sequential(
    0.034 GMac, 0.322% MACs, 
    (0): Conv2d(0.034 GMac, 0.320% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 GMac, 0.001% MACs, inplace=True)
  )
  (dec4): FuseBlock(
    0.04 GMac, 0.385% MACs, 
    (conv1): Conv2d(0.002 GMac, 0.020% MACs, 128, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(0.0 GMac, 0.000% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(0.0 GMac, 0.001% MACs, inplace=True)
    (dconv): ConvTranspose2d(0.0 GMac, 0.003% MACs, 32, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
    (conv2): Conv2d(0.038 GMac, 0.360% MACs, 64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(0.0 GMac, 0.001% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (dec3): FuseBlock(
    0.056 GMac, 0.532% MACs, 
    (conv1): Conv2d(0.001 GMac, 0.012% MACs, 32, 19, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(0.0 GMac, 0.001% MACs, 19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(0.0 GMac, 0.002% MACs, inplace=True)
    (dconv): ConvTranspose2d(0.001 GMac, 0.006% MACs, 19, 19, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), groups=19, bias=False)
    (conv2): Conv2d(0.053 GMac, 0.508% MACs, 38, 19, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(0.0 GMac, 0.003% MACs, 19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (score): Sequential(
    0.027 GMac, 0.259% MACs, 
    (0): Conv2d(0.027 GMac, 0.254% MACs, 19, 19, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 GMac, 0.003% MACs, 19, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 GMac, 0.001% MACs, inplace=True)
  )
  (score_u8): ConvTranspose2d(0.04 GMac, 0.380% MACs, 19, 19, kernel_size=(16, 16), stride=(8, 8), padding=(4, 4), groups=19, bias=False)
)

train_time 1.63
Remaining training time = 105 hour 35 minutes 32 seconds
train_time 1.56
Remaining training time = 102 hour 23 minutes 19 seconds

defaultdict(<class 'float'>, {'batchnorm': 0.115938304, 'conv': 1.840300032})
14439237504.0 14387323
33877827072.0 14387323.0

'''

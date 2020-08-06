##################################################################################
#ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time
#Paper-Link: https://arxiv.org/abs/1805.04554
##################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



__all__ = ["ContextNet"]

class Custom_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthSepConv(nn.Module):
    '''
    We omit the nonlinear-ity between depth-wise and point-wise convolutions in our full resolution branch
    '''
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),  #此处与原文不同
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            Custom_Conv(in_channels, in_channels * t, 1),
            DepthConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out       #输出不含relu激活
        return out



    
class Shallow_net(nn.Module):
    '''
    The ﬁrst layer uses standard convolution while all other layers
use depth-wise separable convolutions with kernel size 3 × 3. The stride is 2 for
all but the last layer, where it is 1.
    '''
    def __init__(self, dw_channels1=32, dw_channels2=64, out_channels=128, **kwargs):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(3, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)


    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x

class Deep_net(nn.Module):

    def __init__(self, in_channels, block_channels,
                 t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks

        self.conv_ = Custom_Conv(3, in_channels, 3, 2)
        self.bottleneck1 = self._layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t[0], 1)
        self.bottleneck2 = self._layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t[1], 1)
        self.bottleneck3 = self._layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t[2], 2)
        self.bottleneck4 = self._layer(LinearBottleneck, block_channels[2], block_channels[3], num_blocks[3], t[3], 2)
        self.bottleneck5 = self._layer(LinearBottleneck, block_channels[3], block_channels[4], num_blocks[4], t[4], 1)
        self.bottleneck6 = self._layer(LinearBottleneck, block_channels[4], block_channels[5], num_blocks[5], t[5], 1)
        # 收尾部分缺少一个conv2d

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)     #原文为DWConv (dilation 4) 3/1, f
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(lower_res_feature, size=(h,w), mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x



class ContextNet(nn.Module):
    def __init__(self, classes, aux=False, **kwargs):
        super(ContextNet, self).__init__()
        self.aux = aux
        self.spatial_detail = Shallow_net(32, 64, 128)
        self.context_feature_extractor = Deep_net(32, [32, 32, 48, 64, 96, 128], [1, 6, 6, 6, 6, 6], [1, 1, 3, 3, 2, 2])
        self.feature_fusion = FeatureFusionModule(128, 128, 128)
        self.classifier = Classifer(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]

        higher_res_features = self.spatial_detail(x)

        x_low = F.interpolate(x, scale_factor = 0.25, mode='bilinear', align_corners=True)

        x = self.context_feature_extractor(x_low)

        x = self.feature_fusion(higher_res_features, x)

        x = self.classifier(x)

        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return x

        # return tuple(outputs)



"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextNet(classes=19).to(device)
    summary(model,(3,512,1024))

    from fvcore.nn.flop_count import flop_count  # https://github.com/facebookresearch/fvcore
    from tools.flops_counter.ptflops import get_model_complexity_info
    from thop import profile  # https://github.com/Lyken17/pytorch-OpCounter

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
            Conv2d-1         [-1, 32, 255, 511]             864
       BatchNorm2d-2         [-1, 32, 255, 511]              64
              ReLU-3         [-1, 32, 255, 511]               0       
       Custom_Conv-4         [-1, 32, 255, 511]               0
       
            Conv2d-5         [-1, 32, 128, 256]             288
       BatchNorm2d-6         [-1, 32, 128, 256]              64
              ReLU-7         [-1, 32, 128, 256]               0
            Conv2d-8         [-1, 64, 128, 256]           2,048
       BatchNorm2d-9         [-1, 64, 128, 256]             128
             ReLU-10         [-1, 64, 128, 256]               0
     DepthSepConv-11         [-1, 64, 128, 256]               0
     
           Conv2d-12          [-1, 64, 64, 128]             576
      BatchNorm2d-13          [-1, 64, 64, 128]             128
             ReLU-14          [-1, 64, 64, 128]               0
           Conv2d-15         [-1, 128, 64, 128]           8,192
      BatchNorm2d-16         [-1, 128, 64, 128]             256
             ReLU-17         [-1, 128, 64, 128]               0
     DepthSepConv-18         [-1, 128, 64, 128]               0
     
           Conv2d-19         [-1, 128, 64, 128]           1,152
      BatchNorm2d-20         [-1, 128, 64, 128]             256
             ReLU-21         [-1, 128, 64, 128]               0
           Conv2d-22         [-1, 128, 64, 128]          16,384
      BatchNorm2d-23         [-1, 128, 64, 128]             256
             ReLU-24         [-1, 128, 64, 128]               0
     DepthSepConv-25         [-1, 128, 64, 128]               0
     
      Shallow_net-26         [-1, 128, 64, 128]               0
     
           Conv2d-27          [-1, 32, 63, 127]             864
      BatchNorm2d-28          [-1, 32, 63, 127]              64
             ReLU-29          [-1, 32, 63, 127]               0
      Custom_Conv-30          [-1, 32, 63, 127]               0
      
           Conv2d-31          [-1, 32, 63, 127]           1,024
      BatchNorm2d-32          [-1, 32, 63, 127]              64
             ReLU-33          [-1, 32, 63, 127]               0
      Custom_Conv-34          [-1, 32, 63, 127]               0
   
           Conv2d-35          [-1, 32, 63, 127]             288
      BatchNorm2d-36          [-1, 32, 63, 127]              64
             ReLU-37          [-1, 32, 63, 127]               0
        DepthConv-38          [-1, 32, 63, 127]               0
           Conv2d-39          [-1, 32, 63, 127]           1,024
      BatchNorm2d-40          [-1, 32, 63, 127]              64
 LinearBottleneck-41          [-1, 32, 63, 127]               0
 
           Conv2d-42         [-1, 192, 63, 127]           6,144
      BatchNorm2d-43         [-1, 192, 63, 127]             384
             ReLU-44         [-1, 192, 63, 127]               0
      Custom_Conv-45         [-1, 192, 63, 127]               0
 
           Conv2d-46         [-1, 192, 63, 127]           1,728
      BatchNorm2d-47         [-1, 192, 63, 127]             384
             ReLU-48         [-1, 192, 63, 127]               0
        DepthConv-49         [-1, 192, 63, 127]               0
           Conv2d-50          [-1, 32, 63, 127]           6,144
      BatchNorm2d-51          [-1, 32, 63, 127]              64
 LinearBottleneck-52          [-1, 32, 63, 127]               0
 
           Conv2d-53         [-1, 192, 63, 127]           6,144
      BatchNorm2d-54         [-1, 192, 63, 127]             384
             ReLU-55         [-1, 192, 63, 127]               0
      Custom_Conv-56         [-1, 192, 63, 127]               0
           Conv2d-57          [-1, 192, 32, 64]           1,728
      BatchNorm2d-58          [-1, 192, 32, 64]             384
             ReLU-59          [-1, 192, 32, 64]               0
        DepthConv-60          [-1, 192, 32, 64]               0
           Conv2d-61           [-1, 48, 32, 64]           9,216
      BatchNorm2d-62           [-1, 48, 32, 64]              96
 LinearBottleneck-63           [-1, 48, 32, 64]               0
           Conv2d-64          [-1, 288, 32, 64]          13,824
      BatchNorm2d-65          [-1, 288, 32, 64]             576
             ReLU-66          [-1, 288, 32, 64]               0
      Custom_Conv-67          [-1, 288, 32, 64]               0
           Conv2d-68          [-1, 288, 32, 64]           2,592
      BatchNorm2d-69          [-1, 288, 32, 64]             576
             ReLU-70          [-1, 288, 32, 64]               0
        DepthConv-71          [-1, 288, 32, 64]               0
           Conv2d-72           [-1, 48, 32, 64]          13,824
      BatchNorm2d-73           [-1, 48, 32, 64]              96
 LinearBottleneck-74           [-1, 48, 32, 64]               0
           Conv2d-75          [-1, 288, 32, 64]          13,824
      BatchNorm2d-76          [-1, 288, 32, 64]             576
             ReLU-77          [-1, 288, 32, 64]               0
      Custom_Conv-78          [-1, 288, 32, 64]               0
           Conv2d-79          [-1, 288, 32, 64]           2,592
      BatchNorm2d-80          [-1, 288, 32, 64]             576
             ReLU-81          [-1, 288, 32, 64]               0
        DepthConv-82          [-1, 288, 32, 64]               0
           Conv2d-83           [-1, 48, 32, 64]          13,824
      BatchNorm2d-84           [-1, 48, 32, 64]              96
 LinearBottleneck-85           [-1, 48, 32, 64]               0
           Conv2d-86          [-1, 288, 32, 64]          13,824
      BatchNorm2d-87          [-1, 288, 32, 64]             576
             ReLU-88          [-1, 288, 32, 64]               0
      Custom_Conv-89          [-1, 288, 32, 64]               0
           Conv2d-90          [-1, 288, 16, 32]           2,592
      BatchNorm2d-91          [-1, 288, 16, 32]             576
             ReLU-92          [-1, 288, 16, 32]               0
        DepthConv-93          [-1, 288, 16, 32]               0
           Conv2d-94           [-1, 64, 16, 32]          18,432
      BatchNorm2d-95           [-1, 64, 16, 32]             128
 LinearBottleneck-96           [-1, 64, 16, 32]               0
           Conv2d-97          [-1, 384, 16, 32]          24,576
      BatchNorm2d-98          [-1, 384, 16, 32]             768
             ReLU-99          [-1, 384, 16, 32]               0
     Custom_Conv-100          [-1, 384, 16, 32]               0
          Conv2d-101          [-1, 384, 16, 32]           3,456
     BatchNorm2d-102          [-1, 384, 16, 32]             768
            ReLU-103          [-1, 384, 16, 32]               0
       DepthConv-104          [-1, 384, 16, 32]               0
          Conv2d-105           [-1, 64, 16, 32]          24,576
     BatchNorm2d-106           [-1, 64, 16, 32]             128
LinearBottleneck-107           [-1, 64, 16, 32]               0
          Conv2d-108          [-1, 384, 16, 32]          24,576
     BatchNorm2d-109          [-1, 384, 16, 32]             768
            ReLU-110          [-1, 384, 16, 32]               0
     Custom_Conv-111          [-1, 384, 16, 32]               0
          Conv2d-112          [-1, 384, 16, 32]           3,456
     BatchNorm2d-113          [-1, 384, 16, 32]             768
            ReLU-114          [-1, 384, 16, 32]               0
       DepthConv-115          [-1, 384, 16, 32]               0
          Conv2d-116           [-1, 64, 16, 32]          24,576
     BatchNorm2d-117           [-1, 64, 16, 32]             128
LinearBottleneck-118           [-1, 64, 16, 32]               0
          Conv2d-119          [-1, 384, 16, 32]          24,576
     BatchNorm2d-120          [-1, 384, 16, 32]             768
            ReLU-121          [-1, 384, 16, 32]               0
     Custom_Conv-122          [-1, 384, 16, 32]               0
          Conv2d-123          [-1, 384, 16, 32]           3,456
     BatchNorm2d-124          [-1, 384, 16, 32]             768
            ReLU-125          [-1, 384, 16, 32]               0
       DepthConv-126          [-1, 384, 16, 32]               0
          Conv2d-127           [-1, 96, 16, 32]          36,864
     BatchNorm2d-128           [-1, 96, 16, 32]             192
LinearBottleneck-129           [-1, 96, 16, 32]               0
          Conv2d-130          [-1, 576, 16, 32]          55,296
     BatchNorm2d-131          [-1, 576, 16, 32]           1,152
            ReLU-132          [-1, 576, 16, 32]               0
     Custom_Conv-133          [-1, 576, 16, 32]               0
          Conv2d-134          [-1, 576, 16, 32]           5,184
     BatchNorm2d-135          [-1, 576, 16, 32]           1,152
            ReLU-136          [-1, 576, 16, 32]               0
       DepthConv-137          [-1, 576, 16, 32]               0
          Conv2d-138           [-1, 96, 16, 32]          55,296
     BatchNorm2d-139           [-1, 96, 16, 32]             192
LinearBottleneck-140           [-1, 96, 16, 32]               0
          Conv2d-141          [-1, 576, 16, 32]          55,296
     BatchNorm2d-142          [-1, 576, 16, 32]           1,152
            ReLU-143          [-1, 576, 16, 32]               0
     Custom_Conv-144          [-1, 576, 16, 32]               0
          Conv2d-145          [-1, 576, 16, 32]           5,184
     BatchNorm2d-146          [-1, 576, 16, 32]           1,152
            ReLU-147          [-1, 576, 16, 32]               0
       DepthConv-148          [-1, 576, 16, 32]               0
          Conv2d-149          [-1, 128, 16, 32]          73,728
     BatchNorm2d-150          [-1, 128, 16, 32]             256
LinearBottleneck-151          [-1, 128, 16, 32]               0
          Conv2d-152          [-1, 768, 16, 32]          98,304
     BatchNorm2d-153          [-1, 768, 16, 32]           1,536
            ReLU-154          [-1, 768, 16, 32]               0
     Custom_Conv-155          [-1, 768, 16, 32]               0
          Conv2d-156          [-1, 768, 16, 32]           6,912
     BatchNorm2d-157          [-1, 768, 16, 32]           1,536
            ReLU-158          [-1, 768, 16, 32]               0
       DepthConv-159          [-1, 768, 16, 32]               0
          Conv2d-160          [-1, 128, 16, 32]          98,304
     BatchNorm2d-161          [-1, 128, 16, 32]             256
LinearBottleneck-162          [-1, 128, 16, 32]               0
        Deep_net-163          [-1, 128, 16, 32]               0
          Conv2d-164         [-1, 128, 64, 128]           1,152
     BatchNorm2d-165         [-1, 128, 64, 128]             256
            ReLU-166         [-1, 128, 64, 128]               0
       DepthConv-167         [-1, 128, 64, 128]               0
          Conv2d-168         [-1, 128, 64, 128]          16,512
     BatchNorm2d-169         [-1, 128, 64, 128]             256
          Conv2d-170         [-1, 128, 64, 128]          16,512
     BatchNorm2d-171         [-1, 128, 64, 128]             256
            ReLU-172         [-1, 128, 64, 128]               0
FeatureFusionModule-173         [-1, 128, 64, 128]               0

          Conv2d-174         [-1, 128, 64, 128]           1,152
     BatchNorm2d-175         [-1, 128, 64, 128]             256
            ReLU-176         [-1, 128, 64, 128]               0
          Conv2d-177         [-1, 128, 64, 128]          16,384
     BatchNorm2d-178         [-1, 128, 64, 128]             256
            ReLU-179         [-1, 128, 64, 128]               0
    DepthSepConv-180         [-1, 128, 64, 128]               0
          Conv2d-181         [-1, 128, 64, 128]           1,152
     BatchNorm2d-182         [-1, 128, 64, 128]             256
            ReLU-183         [-1, 128, 64, 128]               0
          Conv2d-184         [-1, 128, 64, 128]          16,384
     BatchNorm2d-185         [-1, 128, 64, 128]             256
            ReLU-186         [-1, 128, 64, 128]               0
    DepthSepConv-187         [-1, 128, 64, 128]               0
         Dropout-188         [-1, 128, 64, 128]               0
          Conv2d-189          [-1, 19, 64, 128]           2,451
       Classifer-190          [-1, 19, 64, 128]               0
       
          Conv2d-191          [-1, 32, 64, 128]          36,864
     BatchNorm2d-192          [-1, 32, 64, 128]              64
            ReLU-193          [-1, 32, 64, 128]               0
         Dropout-194          [-1, 32, 64, 128]               0
          Conv2d-195          [-1, 19, 64, 128]             627
================================================================
Total params: 914,118
Trainable params: 914,118
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 6.00
Forward/backward pass size (MB): 927.87
Params size (MB): 3.49
Estimated Total Size (MB): 937.35
----------------------------------------------------------------
Skipped operation aten::relu_ 38 time(s)
Skipped operation aten::mul 2 time(s)
Skipped operation aten::upsample_bilinear2d 3 time(s)
Skipped operation aten::add 9 time(s)
Skipped operation aten::dropout 1 time(s)
ContextNet(
  2.084 GMac, 100.000% MACs, 
  (spatial_detail): Shallow_net(
    0.438 GMac, 21.000% MACs, 
    (conv): Custom_Conv(
      0.125 GMac, 6.004% MACs, 
      (conv): Sequential(
        0.125 GMac, 6.004% MACs, 
        (0): Conv2d(0.113 GMac, 5.403% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.008 GMac, 0.400% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.004 GMac, 0.200% MACs, inplace=True)
      )
    )
    (dsconv1): DepthSepConv(
      0.086 GMac, 4.127% MACs, 
      (conv): Sequential(
        0.086 GMac, 4.127% MACs, 
        (0): Conv2d(0.009 GMac, 0.453% MACs, 32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
        (3): Conv2d(0.067 GMac, 3.221% MACs, 32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.004 GMac, 0.201% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(0.002 GMac, 0.101% MACs, inplace=True)
      )
    )
    (dsconv2): DepthSepConv(
      0.077 GMac, 3.674% MACs, 
      (conv): Sequential(
        0.077 GMac, 3.674% MACs, 
        (0): Conv2d(0.005 GMac, 0.226% MACs, 64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.050% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.025% MACs, inplace=True)
        (3): Conv2d(0.067 GMac, 3.221% MACs, 64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
      )
    )
    (dsconv3): DepthSepConv(
      0.15 GMac, 7.196% MACs, 
      (conv): Sequential(
        0.15 GMac, 7.196% MACs, 
        (0): Conv2d(0.009 GMac, 0.453% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
        (3): Conv2d(0.134 GMac, 6.441% MACs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
      )
    )
  )
  (context_feature_extractor): Deep_net(
    0.73 GMac, 35.027% MACs, 
    (conv_): Custom_Conv(
      0.008 GMac, 0.369% MACs, 
      (conv): Sequential(
        0.008 GMac, 0.369% MACs, 
        (0): Conv2d(0.007 GMac, 0.332% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (1): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.0 GMac, 0.012% MACs, inplace=True)
      )
    )
    (bottleneck1): Sequential(
      0.021 GMac, 0.995% MACs, 
      (0): LinearBottleneck(
        0.021 GMac, 0.995% MACs, 
        (block): Sequential(
          0.021 GMac, 0.995% MACs, 
          (0): Custom_Conv(
            0.009 GMac, 0.430% MACs, 
            (conv): Sequential(
              0.009 GMac, 0.430% MACs, 
              (0): Conv2d(0.008 GMac, 0.393% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.012% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.003 GMac, 0.147% MACs, 
            (conv): Sequential(
              0.003 GMac, 0.147% MACs, 
              (0): Conv2d(0.002 GMac, 0.111% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.012% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.008 GMac, 0.393% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (bottleneck2): Sequential(
      0.122 GMac, 5.849% MACs, 
      (0): LinearBottleneck(
        0.122 GMac, 5.849% MACs, 
        (block): Sequential(
          0.122 GMac, 5.849% MACs, 
          (0): Custom_Conv(
            0.054 GMac, 2.580% MACs, 
            (conv): Sequential(
              0.054 GMac, 2.580% MACs, 
              (0): Conv2d(0.049 GMac, 2.359% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.003 GMac, 0.147% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.002 GMac, 0.074% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.018 GMac, 0.885% MACs, 
            (conv): Sequential(
              0.018 GMac, 0.885% MACs, 
              (0): Conv2d(0.014 GMac, 0.664% MACs, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(0.003 GMac, 0.147% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.002 GMac, 0.074% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.049 GMac, 2.359% MACs, 192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (bottleneck3): Sequential(
      0.209 GMac, 10.025% MACs, 
      (0): LinearBottleneck(
        0.078 GMac, 3.722% MACs, 
        (block): Sequential(
          0.078 GMac, 3.722% MACs, 
          (0): Custom_Conv(
            0.054 GMac, 2.580% MACs, 
            (conv): Sequential(
              0.054 GMac, 2.580% MACs, 
              (0): Conv2d(0.049 GMac, 2.359% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.003 GMac, 0.147% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.002 GMac, 0.074% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.005 GMac, 0.226% MACs, 
            (conv): Sequential(
              0.005 GMac, 0.226% MACs, 
              (0): Conv2d(0.004 GMac, 0.170% MACs, 192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.038% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.019% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.019 GMac, 0.906% MACs, 192, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.009% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): LinearBottleneck(
        0.066 GMac, 3.152% MACs, 
        (block): Sequential(
          0.066 GMac, 3.152% MACs, 
          (0): Custom_Conv(
            0.03 GMac, 1.444% MACs, 
            (conv): Sequential(
              0.03 GMac, 1.444% MACs, 
              (0): Conv2d(0.028 GMac, 1.359% MACs, 48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.057% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.001 GMac, 0.028% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.007 GMac, 0.340% MACs, 
            (conv): Sequential(
              0.007 GMac, 0.340% MACs, 
              (0): Conv2d(0.005 GMac, 0.255% MACs, 288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.057% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.001 GMac, 0.028% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.028 GMac, 1.359% MACs, 288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.009% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (2): LinearBottleneck(
        0.066 GMac, 3.152% MACs, 
        (block): Sequential(
          0.066 GMac, 3.152% MACs, 
          (0): Custom_Conv(
            0.03 GMac, 1.444% MACs, 
            (conv): Sequential(
              0.03 GMac, 1.444% MACs, 
              (0): Conv2d(0.028 GMac, 1.359% MACs, 48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.057% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.001 GMac, 0.028% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.007 GMac, 0.340% MACs, 
            (conv): Sequential(
              0.007 GMac, 0.340% MACs, 
              (0): Conv2d(0.005 GMac, 0.255% MACs, 288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.057% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.001 GMac, 0.028% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.028 GMac, 1.359% MACs, 288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.009% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (bottleneck4): Sequential(
      0.098 GMac, 4.690% MACs, 
      (0): LinearBottleneck(
        0.041 GMac, 1.985% MACs, 
        (block): Sequential(
          0.041 GMac, 1.985% MACs, 
          (0): Custom_Conv(
            0.03 GMac, 1.444% MACs, 
            (conv): Sequential(
              0.03 GMac, 1.444% MACs, 
              (0): Conv2d(0.028 GMac, 1.359% MACs, 48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.057% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.001 GMac, 0.028% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.002 GMac, 0.085% MACs, 
            (conv): Sequential(
              0.002 GMac, 0.085% MACs, 
              (0): Conv2d(0.001 GMac, 0.064% MACs, 288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=288, bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.014% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.007% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.009 GMac, 0.453% MACs, 288, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): LinearBottleneck(
        0.028 GMac, 1.352% MACs, 
        (block): Sequential(
          0.028 GMac, 1.352% MACs, 
          (0): Custom_Conv(
            0.013 GMac, 0.632% MACs, 
            (conv): Sequential(
              0.013 GMac, 0.632% MACs, 
              (0): Conv2d(0.013 GMac, 0.604% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.002 GMac, 0.113% MACs, 
            (conv): Sequential(
              0.002 GMac, 0.113% MACs, 
              (0): Conv2d(0.002 GMac, 0.085% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.013 GMac, 0.604% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (2): LinearBottleneck(
        0.028 GMac, 1.352% MACs, 
        (block): Sequential(
          0.028 GMac, 1.352% MACs, 
          (0): Custom_Conv(
            0.013 GMac, 0.632% MACs, 
            (conv): Sequential(
              0.013 GMac, 0.632% MACs, 
              (0): Conv2d(0.013 GMac, 0.604% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.002 GMac, 0.113% MACs, 
            (conv): Sequential(
              0.002 GMac, 0.113% MACs, 
              (0): Conv2d(0.002 GMac, 0.085% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.013 GMac, 0.604% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (bottleneck5): Sequential(
      0.096 GMac, 4.590% MACs, 
      (0): LinearBottleneck(
        0.035 GMac, 1.656% MACs, 
        (block): Sequential(
          0.035 GMac, 1.656% MACs, 
          (0): Custom_Conv(
            0.013 GMac, 0.632% MACs, 
            (conv): Sequential(
              0.013 GMac, 0.632% MACs, 
              (0): Conv2d(0.013 GMac, 0.604% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.002 GMac, 0.113% MACs, 
            (conv): Sequential(
              0.002 GMac, 0.113% MACs, 
              (0): Conv2d(0.002 GMac, 0.085% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(0.0 GMac, 0.019% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.009% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.019 GMac, 0.906% MACs, 384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.005% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): LinearBottleneck(
        0.061 GMac, 2.934% MACs, 
        (block): Sequential(
          0.061 GMac, 2.934% MACs, 
          (0): Custom_Conv(
            0.029 GMac, 1.401% MACs, 
            (conv): Sequential(
              0.029 GMac, 1.401% MACs, 
              (0): Conv2d(0.028 GMac, 1.359% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.028% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.014% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.004 GMac, 0.170% MACs, 
            (conv): Sequential(
              0.004 GMac, 0.170% MACs, 
              (0): Conv2d(0.003 GMac, 0.127% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.028% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.014% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.028 GMac, 1.359% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.005% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (bottleneck6): Sequential(
      0.177 GMac, 8.509% MACs, 
      (0): LinearBottleneck(
        0.071 GMac, 3.389% MACs, 
        (block): Sequential(
          0.071 GMac, 3.389% MACs, 
          (0): Custom_Conv(
            0.029 GMac, 1.401% MACs, 
            (conv): Sequential(
              0.029 GMac, 1.401% MACs, 
              (0): Conv2d(0.028 GMac, 1.359% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.028% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.014% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.004 GMac, 0.170% MACs, 
            (conv): Sequential(
              0.004 GMac, 0.170% MACs, 
              (0): Conv2d(0.003 GMac, 0.127% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.028% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.014% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.038 GMac, 1.812% MACs, 576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.006% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): LinearBottleneck(
        0.107 GMac, 5.120% MACs, 
        (block): Sequential(
          0.107 GMac, 5.120% MACs, 
          (0): Custom_Conv(
            0.052 GMac, 2.472% MACs, 
            (conv): Sequential(
              0.052 GMac, 2.472% MACs, 
              (0): Conv2d(0.05 GMac, 2.416% MACs, 128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.038% MACs, 768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.019% MACs, inplace=True)
            )
          )
          (1): DepthConv(
            0.005 GMac, 0.226% MACs, 
            (conv): Sequential(
              0.005 GMac, 0.226% MACs, 
              (0): Conv2d(0.004 GMac, 0.170% MACs, 768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (1): BatchNorm2d(0.001 GMac, 0.038% MACs, 768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(0.0 GMac, 0.019% MACs, inplace=True)
            )
          )
          (2): Conv2d(0.05 GMac, 2.416% MACs, 768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(0.0 GMac, 0.006% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
  )
  (feature_fusion): FeatureFusionModule(
    0.288 GMac, 13.839% MACs, 
    (dwconv): DepthConv(
      0.013 GMac, 0.604% MACs, 
      (conv): Sequential(
        0.013 GMac, 0.604% MACs, 
        (0): Conv2d(0.009 GMac, 0.453% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
      )
    )
    (conv_lower_res): Sequential(
      0.137 GMac, 6.592% MACs, 
      (0): Conv2d(0.135 GMac, 6.492% MACs, 128, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (conv_higher_res): Sequential(
      0.137 GMac, 6.592% MACs, 
      (0): Conv2d(0.135 GMac, 6.492% MACs, 128, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (relu): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
  )
  (classifier): Classifer(
    0.32 GMac, 15.356% MACs, 
    (dsconv1): DepthSepConv(
      0.15 GMac, 7.196% MACs, 
      (conv): Sequential(
        0.15 GMac, 7.196% MACs, 
        (0): Conv2d(0.009 GMac, 0.453% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
        (3): Conv2d(0.134 GMac, 6.441% MACs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
      )
    )
    (dsconv2): DepthSepConv(
      0.15 GMac, 7.196% MACs, 
      (conv): Sequential(
        0.15 GMac, 7.196% MACs, 
        (0): Conv2d(0.009 GMac, 0.453% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (1): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
        (3): Conv2d(0.134 GMac, 6.441% MACs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.002 GMac, 0.101% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(0.001 GMac, 0.050% MACs, inplace=True)
      )
    )
    (conv): Sequential(
      0.02 GMac, 0.964% MACs, 
      (0): Dropout(0.0 GMac, 0.000% MACs, p=0.1, inplace=False)
      (1): Conv2d(0.02 GMac, 0.964% MACs, 128, 19, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (auxlayer): Sequential(
    0.308 GMac, 14.777% MACs, 
    (0): Conv2d(0.302 GMac, 14.493% MACs, 128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.001 GMac, 0.025% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0.0 GMac, 0.013% MACs, inplace=True)
    (3): Dropout(0.0 GMac, 0.000% MACs, p=0.1, inplace=False)
    (4): Conv2d(0.005 GMac, 0.247% MACs, 32, 19, kernel_size=(1, 1), stride=(1, 1))
  )
)

train_time :1.433
val_time 0.168
    '''

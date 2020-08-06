##################################################################################
#ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
#Paper-Link:  https://arxiv.org/pdf/1606.02147.pdf
##################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["ENet"]


class InitialBlock(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size, padding=0, bias=False,relu=True):
        super(InitialBlock, self).__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels-3,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias,
        )
        # MP need padding too
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation


    def forward(self, input):
        main = self.main_branch(input)
        ext = self.ext_branch(input)

        out = torch.cat((main, ext), dim=1)

        out = self.batch_norm(out)
        return self.out_prelu(out)

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0,
                 dilation=1, asymmetric=False, dropout_prob=0., bias=False, relu=True):
        super(RegularBottleneck, self).__init__()

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 1x1 projection conv
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation,
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size,1),
                          stride=1, padding=(padding,0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1,kernel_size),
                          stride=1, padding=(0, padding), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels),
                activation,
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(channels),
            activation,
        )
        self.ext_regu1 = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, input):
         main = input

         ext = self.ext_conv1(input)
         ext = self.ext_conv2(ext)
         ext = self.ext_conv3(ext)
         ext = self.ext_regu1(ext)

         out = main + ext
         return self.out_prelu(out)

class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 return_indices=False,
                 dropout_prob=0.,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2, no padding
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels,internal_channels,kernel_size=2,stride=2,bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        # calculate for padding ch_ext - ch_main
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate, padding for less channels of main branch
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices

class UpsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0.,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)

class ENet(nn.Module):
    def __init__(self, classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        # source code
        self.name='BaseLine_ENet_trans'

        self.initial_block = InitialBlock(3, 16, kernel_size=3 ,padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            padding=1,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            padding=1,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        self.project_layer = nn.Conv2d(128, classes, 1, bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        #x = self.project_layer(x)
        #x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)


        return x


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ENet(classes=19).to(device)
    summary(model,(3,512,1024))
    # print(model)
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 13, 256, 512]             351
         MaxPool2d-2          [-1, 3, 256, 512]               0
       BatchNorm2d-3         [-1, 16, 256, 512]              32
             PReLU-4         [-1, 16, 256, 512]               1
      InitialBlock-5         [-1, 16, 256, 512]               0
         MaxPool2d-6  [[-1, 16, 128, 256], [-1, 16, 128, 256]]               0
            Conv2d-7          [-1, 4, 128, 256]             256
       BatchNorm2d-8          [-1, 4, 128, 256]               8
             PReLU-9          [-1, 4, 128, 256]               1
            PReLU-10          [-1, 4, 128, 256]               1
            PReLU-11          [-1, 4, 128, 256]               1
            PReLU-12          [-1, 4, 128, 256]               1
           Conv2d-13          [-1, 4, 128, 256]             144
      BatchNorm2d-14          [-1, 4, 128, 256]               8
            PReLU-15          [-1, 4, 128, 256]               1
            PReLU-16          [-1, 4, 128, 256]               1
            PReLU-17          [-1, 4, 128, 256]               1
            PReLU-18          [-1, 4, 128, 256]               1
           Conv2d-19         [-1, 64, 128, 256]             256
      BatchNorm2d-20         [-1, 64, 128, 256]             128
            PReLU-21         [-1, 64, 128, 256]               1
            PReLU-22         [-1, 64, 128, 256]               1
            PReLU-23         [-1, 64, 128, 256]               1
            PReLU-24         [-1, 64, 128, 256]               1
        Dropout2d-25         [-1, 64, 128, 256]               0
            PReLU-26         [-1, 64, 128, 256]               1
            PReLU-27         [-1, 64, 128, 256]               1
            PReLU-28         [-1, 64, 128, 256]               1
            PReLU-29         [-1, 64, 128, 256]               1
DownsamplingBottleneck-30  [[-1, 64, 128, 256], [-1, 16, 128, 256]]               0
           Conv2d-31         [-1, 16, 128, 256]           1,024
      BatchNorm2d-32         [-1, 16, 128, 256]              32
            PReLU-33         [-1, 16, 128, 256]               1
            PReLU-34         [-1, 16, 128, 256]               1
            PReLU-35         [-1, 16, 128, 256]               1
            PReLU-36         [-1, 16, 128, 256]               1
           Conv2d-37         [-1, 16, 128, 256]           2,304
      BatchNorm2d-38         [-1, 16, 128, 256]              32
            PReLU-39         [-1, 16, 128, 256]               1
            PReLU-40         [-1, 16, 128, 256]               1
            PReLU-41         [-1, 16, 128, 256]               1
            PReLU-42         [-1, 16, 128, 256]               1
           Conv2d-43         [-1, 64, 128, 256]           1,024
      BatchNorm2d-44         [-1, 64, 128, 256]             128
            PReLU-45         [-1, 64, 128, 256]               1
            PReLU-46         [-1, 64, 128, 256]               1
            PReLU-47         [-1, 64, 128, 256]               1
            PReLU-48         [-1, 64, 128, 256]               1
        Dropout2d-49         [-1, 64, 128, 256]               0
            PReLU-50         [-1, 64, 128, 256]               1
            PReLU-51         [-1, 64, 128, 256]               1
            PReLU-52         [-1, 64, 128, 256]               1
            PReLU-53         [-1, 64, 128, 256]               1
RegularBottleneck-54         [-1, 64, 128, 256]               0
           Conv2d-55         [-1, 16, 128, 256]           1,024
      BatchNorm2d-56         [-1, 16, 128, 256]              32
            PReLU-57         [-1, 16, 128, 256]               1
            PReLU-58         [-1, 16, 128, 256]               1
            PReLU-59         [-1, 16, 128, 256]               1
            PReLU-60         [-1, 16, 128, 256]               1
           Conv2d-61         [-1, 16, 128, 256]           2,304
      BatchNorm2d-62         [-1, 16, 128, 256]              32
            PReLU-63         [-1, 16, 128, 256]               1
            PReLU-64         [-1, 16, 128, 256]               1
            PReLU-65         [-1, 16, 128, 256]               1
            PReLU-66         [-1, 16, 128, 256]               1
           Conv2d-67         [-1, 64, 128, 256]           1,024
      BatchNorm2d-68         [-1, 64, 128, 256]             128
            PReLU-69         [-1, 64, 128, 256]               1
            PReLU-70         [-1, 64, 128, 256]               1
            PReLU-71         [-1, 64, 128, 256]               1
            PReLU-72         [-1, 64, 128, 256]               1
        Dropout2d-73         [-1, 64, 128, 256]               0
            PReLU-74         [-1, 64, 128, 256]               1
            PReLU-75         [-1, 64, 128, 256]               1
            PReLU-76         [-1, 64, 128, 256]               1
            PReLU-77         [-1, 64, 128, 256]               1
RegularBottleneck-78         [-1, 64, 128, 256]               0
           Conv2d-79         [-1, 16, 128, 256]           1,024
      BatchNorm2d-80         [-1, 16, 128, 256]              32
            PReLU-81         [-1, 16, 128, 256]               1
            PReLU-82         [-1, 16, 128, 256]               1
            PReLU-83         [-1, 16, 128, 256]               1
            PReLU-84         [-1, 16, 128, 256]               1
           Conv2d-85         [-1, 16, 128, 256]           2,304
      BatchNorm2d-86         [-1, 16, 128, 256]              32
            PReLU-87         [-1, 16, 128, 256]               1
            PReLU-88         [-1, 16, 128, 256]               1
            PReLU-89         [-1, 16, 128, 256]               1
            PReLU-90         [-1, 16, 128, 256]               1
           Conv2d-91         [-1, 64, 128, 256]           1,024
      BatchNorm2d-92         [-1, 64, 128, 256]             128
            PReLU-93         [-1, 64, 128, 256]               1
            PReLU-94         [-1, 64, 128, 256]               1
            PReLU-95         [-1, 64, 128, 256]               1
            PReLU-96         [-1, 64, 128, 256]               1
        Dropout2d-97         [-1, 64, 128, 256]               0
            PReLU-98         [-1, 64, 128, 256]               1
            PReLU-99         [-1, 64, 128, 256]               1
           PReLU-100         [-1, 64, 128, 256]               1
           PReLU-101         [-1, 64, 128, 256]               1
RegularBottleneck-102         [-1, 64, 128, 256]               0
          Conv2d-103         [-1, 16, 128, 256]           1,024
     BatchNorm2d-104         [-1, 16, 128, 256]              32
           PReLU-105         [-1, 16, 128, 256]               1
           PReLU-106         [-1, 16, 128, 256]               1
           PReLU-107         [-1, 16, 128, 256]               1
           PReLU-108         [-1, 16, 128, 256]               1
          Conv2d-109         [-1, 16, 128, 256]           2,304
     BatchNorm2d-110         [-1, 16, 128, 256]              32
           PReLU-111         [-1, 16, 128, 256]               1
           PReLU-112         [-1, 16, 128, 256]               1
           PReLU-113         [-1, 16, 128, 256]               1
           PReLU-114         [-1, 16, 128, 256]               1
          Conv2d-115         [-1, 64, 128, 256]           1,024
     BatchNorm2d-116         [-1, 64, 128, 256]             128
           PReLU-117         [-1, 64, 128, 256]               1
           PReLU-118         [-1, 64, 128, 256]               1
           PReLU-119         [-1, 64, 128, 256]               1
           PReLU-120         [-1, 64, 128, 256]               1
       Dropout2d-121         [-1, 64, 128, 256]               0
           PReLU-122         [-1, 64, 128, 256]               1
           PReLU-123         [-1, 64, 128, 256]               1
           PReLU-124         [-1, 64, 128, 256]               1
           PReLU-125         [-1, 64, 128, 256]               1
RegularBottleneck-126         [-1, 64, 128, 256]               0
       MaxPool2d-127  [[-1, 64, 64, 128], [-1, 64, 64, 128]]               0
          Conv2d-128          [-1, 16, 64, 128]           4,096
     BatchNorm2d-129          [-1, 16, 64, 128]              32
           PReLU-130          [-1, 16, 64, 128]               1
           PReLU-131          [-1, 16, 64, 128]               1
           PReLU-132          [-1, 16, 64, 128]               1
           PReLU-133          [-1, 16, 64, 128]               1
          Conv2d-134          [-1, 16, 64, 128]           2,304
     BatchNorm2d-135          [-1, 16, 64, 128]              32
           PReLU-136          [-1, 16, 64, 128]               1
           PReLU-137          [-1, 16, 64, 128]               1
           PReLU-138          [-1, 16, 64, 128]               1
           PReLU-139          [-1, 16, 64, 128]               1
          Conv2d-140         [-1, 128, 64, 128]           2,048
     BatchNorm2d-141         [-1, 128, 64, 128]             256
           PReLU-142         [-1, 128, 64, 128]               1
           PReLU-143         [-1, 128, 64, 128]               1
           PReLU-144         [-1, 128, 64, 128]               1
           PReLU-145         [-1, 128, 64, 128]               1
       Dropout2d-146         [-1, 128, 64, 128]               0
           PReLU-147         [-1, 128, 64, 128]               1
           PReLU-148         [-1, 128, 64, 128]               1
           PReLU-149         [-1, 128, 64, 128]               1
           PReLU-150         [-1, 128, 64, 128]               1
DownsamplingBottleneck-151  [[-1, 128, 64, 128], [-1, 64, 64, 128]]               0
          Conv2d-152          [-1, 32, 64, 128]           4,096
     BatchNorm2d-153          [-1, 32, 64, 128]              64
           PReLU-154          [-1, 32, 64, 128]               1
           PReLU-155          [-1, 32, 64, 128]               1
           PReLU-156          [-1, 32, 64, 128]               1
           PReLU-157          [-1, 32, 64, 128]               1
          Conv2d-158          [-1, 32, 64, 128]           9,216
     BatchNorm2d-159          [-1, 32, 64, 128]              64
           PReLU-160          [-1, 32, 64, 128]               1
           PReLU-161          [-1, 32, 64, 128]               1
           PReLU-162          [-1, 32, 64, 128]               1
           PReLU-163          [-1, 32, 64, 128]               1
          Conv2d-164         [-1, 128, 64, 128]           4,096
     BatchNorm2d-165         [-1, 128, 64, 128]             256
           PReLU-166         [-1, 128, 64, 128]               1
           PReLU-167         [-1, 128, 64, 128]               1
           PReLU-168         [-1, 128, 64, 128]               1
           PReLU-169         [-1, 128, 64, 128]               1
       Dropout2d-170         [-1, 128, 64, 128]               0
           PReLU-171         [-1, 128, 64, 128]               1
           PReLU-172         [-1, 128, 64, 128]               1
           PReLU-173         [-1, 128, 64, 128]               1
           PReLU-174         [-1, 128, 64, 128]               1
RegularBottleneck-175         [-1, 128, 64, 128]               0
          Conv2d-176          [-1, 32, 64, 128]           4,096
     BatchNorm2d-177          [-1, 32, 64, 128]              64
           PReLU-178          [-1, 32, 64, 128]               1
           PReLU-179          [-1, 32, 64, 128]               1
           PReLU-180          [-1, 32, 64, 128]               1
           PReLU-181          [-1, 32, 64, 128]               1
          Conv2d-182          [-1, 32, 64, 128]           9,216
     BatchNorm2d-183          [-1, 32, 64, 128]              64
           PReLU-184          [-1, 32, 64, 128]               1
           PReLU-185          [-1, 32, 64, 128]               1
           PReLU-186          [-1, 32, 64, 128]               1
           PReLU-187          [-1, 32, 64, 128]               1
          Conv2d-188         [-1, 128, 64, 128]           4,096
     BatchNorm2d-189         [-1, 128, 64, 128]             256
           PReLU-190         [-1, 128, 64, 128]               1
           PReLU-191         [-1, 128, 64, 128]               1
           PReLU-192         [-1, 128, 64, 128]               1
           PReLU-193         [-1, 128, 64, 128]               1
       Dropout2d-194         [-1, 128, 64, 128]               0
           PReLU-195         [-1, 128, 64, 128]               1
           PReLU-196         [-1, 128, 64, 128]               1
           PReLU-197         [-1, 128, 64, 128]               1
           PReLU-198         [-1, 128, 64, 128]               1
RegularBottleneck-199         [-1, 128, 64, 128]               0
          Conv2d-200          [-1, 32, 64, 128]           4,096
     BatchNorm2d-201          [-1, 32, 64, 128]              64
           PReLU-202          [-1, 32, 64, 128]               1
           PReLU-203          [-1, 32, 64, 128]               1
           PReLU-204          [-1, 32, 64, 128]               1
           PReLU-205          [-1, 32, 64, 128]               1
          Conv2d-206          [-1, 32, 64, 128]           5,120
     BatchNorm2d-207          [-1, 32, 64, 128]              64
           PReLU-208          [-1, 32, 64, 128]               1
           PReLU-209          [-1, 32, 64, 128]               1
           PReLU-210          [-1, 32, 64, 128]               1
           PReLU-211          [-1, 32, 64, 128]               1
          Conv2d-212          [-1, 32, 64, 128]           5,120
     BatchNorm2d-213          [-1, 32, 64, 128]              64
           PReLU-214          [-1, 32, 64, 128]               1
           PReLU-215          [-1, 32, 64, 128]               1
           PReLU-216          [-1, 32, 64, 128]               1
           PReLU-217          [-1, 32, 64, 128]               1
          Conv2d-218         [-1, 128, 64, 128]           4,096
     BatchNorm2d-219         [-1, 128, 64, 128]             256
           PReLU-220         [-1, 128, 64, 128]               1
           PReLU-221         [-1, 128, 64, 128]               1
           PReLU-222         [-1, 128, 64, 128]               1
           PReLU-223         [-1, 128, 64, 128]               1
       Dropout2d-224         [-1, 128, 64, 128]               0
           PReLU-225         [-1, 128, 64, 128]               1
           PReLU-226         [-1, 128, 64, 128]               1
           PReLU-227         [-1, 128, 64, 128]               1
           PReLU-228         [-1, 128, 64, 128]               1
RegularBottleneck-229         [-1, 128, 64, 128]               0
          Conv2d-230          [-1, 32, 64, 128]           4,096
     BatchNorm2d-231          [-1, 32, 64, 128]              64
           PReLU-232          [-1, 32, 64, 128]               1
           PReLU-233          [-1, 32, 64, 128]               1
           PReLU-234          [-1, 32, 64, 128]               1
           PReLU-235          [-1, 32, 64, 128]               1
          Conv2d-236          [-1, 32, 64, 128]           9,216
     BatchNorm2d-237          [-1, 32, 64, 128]              64
           PReLU-238          [-1, 32, 64, 128]               1
           PReLU-239          [-1, 32, 64, 128]               1
           PReLU-240          [-1, 32, 64, 128]               1
           PReLU-241          [-1, 32, 64, 128]               1
          Conv2d-242         [-1, 128, 64, 128]           4,096
     BatchNorm2d-243         [-1, 128, 64, 128]             256
           PReLU-244         [-1, 128, 64, 128]               1
           PReLU-245         [-1, 128, 64, 128]               1
           PReLU-246         [-1, 128, 64, 128]               1
           PReLU-247         [-1, 128, 64, 128]               1
       Dropout2d-248         [-1, 128, 64, 128]               0
           PReLU-249         [-1, 128, 64, 128]               1
           PReLU-250         [-1, 128, 64, 128]               1
           PReLU-251         [-1, 128, 64, 128]               1
           PReLU-252         [-1, 128, 64, 128]               1
RegularBottleneck-253         [-1, 128, 64, 128]               0
          Conv2d-254          [-1, 32, 64, 128]           4,096
     BatchNorm2d-255          [-1, 32, 64, 128]              64
           PReLU-256          [-1, 32, 64, 128]               1
           PReLU-257          [-1, 32, 64, 128]               1
           PReLU-258          [-1, 32, 64, 128]               1
           PReLU-259          [-1, 32, 64, 128]               1
          Conv2d-260          [-1, 32, 64, 128]           9,216
     BatchNorm2d-261          [-1, 32, 64, 128]              64
           PReLU-262          [-1, 32, 64, 128]               1
           PReLU-263          [-1, 32, 64, 128]               1
           PReLU-264          [-1, 32, 64, 128]               1
           PReLU-265          [-1, 32, 64, 128]               1
          Conv2d-266         [-1, 128, 64, 128]           4,096
     BatchNorm2d-267         [-1, 128, 64, 128]             256
           PReLU-268         [-1, 128, 64, 128]               1
           PReLU-269         [-1, 128, 64, 128]               1
           PReLU-270         [-1, 128, 64, 128]               1
           PReLU-271         [-1, 128, 64, 128]               1
       Dropout2d-272         [-1, 128, 64, 128]               0
           PReLU-273         [-1, 128, 64, 128]               1
           PReLU-274         [-1, 128, 64, 128]               1
           PReLU-275         [-1, 128, 64, 128]               1
           PReLU-276         [-1, 128, 64, 128]               1
RegularBottleneck-277         [-1, 128, 64, 128]               0
          Conv2d-278          [-1, 32, 64, 128]           4,096
     BatchNorm2d-279          [-1, 32, 64, 128]              64
           PReLU-280          [-1, 32, 64, 128]               1
           PReLU-281          [-1, 32, 64, 128]               1
           PReLU-282          [-1, 32, 64, 128]               1
           PReLU-283          [-1, 32, 64, 128]               1
          Conv2d-284          [-1, 32, 64, 128]           9,216
     BatchNorm2d-285          [-1, 32, 64, 128]              64
           PReLU-286          [-1, 32, 64, 128]               1
           PReLU-287          [-1, 32, 64, 128]               1
           PReLU-288          [-1, 32, 64, 128]               1
           PReLU-289          [-1, 32, 64, 128]               1
          Conv2d-290         [-1, 128, 64, 128]           4,096
     BatchNorm2d-291         [-1, 128, 64, 128]             256
           PReLU-292         [-1, 128, 64, 128]               1
           PReLU-293         [-1, 128, 64, 128]               1
           PReLU-294         [-1, 128, 64, 128]               1
           PReLU-295         [-1, 128, 64, 128]               1
       Dropout2d-296         [-1, 128, 64, 128]               0
           PReLU-297         [-1, 128, 64, 128]               1
           PReLU-298         [-1, 128, 64, 128]               1
           PReLU-299         [-1, 128, 64, 128]               1
           PReLU-300         [-1, 128, 64, 128]               1
RegularBottleneck-301         [-1, 128, 64, 128]               0
          Conv2d-302          [-1, 32, 64, 128]           4,096
     BatchNorm2d-303          [-1, 32, 64, 128]              64
           PReLU-304          [-1, 32, 64, 128]               1
           PReLU-305          [-1, 32, 64, 128]               1
           PReLU-306          [-1, 32, 64, 128]               1
           PReLU-307          [-1, 32, 64, 128]               1
          Conv2d-308          [-1, 32, 64, 128]           5,120
     BatchNorm2d-309          [-1, 32, 64, 128]              64
           PReLU-310          [-1, 32, 64, 128]               1
           PReLU-311          [-1, 32, 64, 128]               1
           PReLU-312          [-1, 32, 64, 128]               1
           PReLU-313          [-1, 32, 64, 128]               1
          Conv2d-314          [-1, 32, 64, 128]           5,120
     BatchNorm2d-315          [-1, 32, 64, 128]              64
           PReLU-316          [-1, 32, 64, 128]               1
           PReLU-317          [-1, 32, 64, 128]               1
           PReLU-318          [-1, 32, 64, 128]               1
           PReLU-319          [-1, 32, 64, 128]               1
          Conv2d-320         [-1, 128, 64, 128]           4,096
     BatchNorm2d-321         [-1, 128, 64, 128]             256
           PReLU-322         [-1, 128, 64, 128]               1
           PReLU-323         [-1, 128, 64, 128]               1
           PReLU-324         [-1, 128, 64, 128]               1
           PReLU-325         [-1, 128, 64, 128]               1
       Dropout2d-326         [-1, 128, 64, 128]               0
           PReLU-327         [-1, 128, 64, 128]               1
           PReLU-328         [-1, 128, 64, 128]               1
           PReLU-329         [-1, 128, 64, 128]               1
           PReLU-330         [-1, 128, 64, 128]               1
RegularBottleneck-331         [-1, 128, 64, 128]               0
          Conv2d-332          [-1, 32, 64, 128]           4,096
     BatchNorm2d-333          [-1, 32, 64, 128]              64
           PReLU-334          [-1, 32, 64, 128]               1
           PReLU-335          [-1, 32, 64, 128]               1
           PReLU-336          [-1, 32, 64, 128]               1
           PReLU-337          [-1, 32, 64, 128]               1
          Conv2d-338          [-1, 32, 64, 128]           9,216
     BatchNorm2d-339          [-1, 32, 64, 128]              64
           PReLU-340          [-1, 32, 64, 128]               1
           PReLU-341          [-1, 32, 64, 128]               1
           PReLU-342          [-1, 32, 64, 128]               1
           PReLU-343          [-1, 32, 64, 128]               1
          Conv2d-344         [-1, 128, 64, 128]           4,096
     BatchNorm2d-345         [-1, 128, 64, 128]             256
           PReLU-346         [-1, 128, 64, 128]               1
           PReLU-347         [-1, 128, 64, 128]               1
           PReLU-348         [-1, 128, 64, 128]               1
           PReLU-349         [-1, 128, 64, 128]               1
       Dropout2d-350         [-1, 128, 64, 128]               0
           PReLU-351         [-1, 128, 64, 128]               1
           PReLU-352         [-1, 128, 64, 128]               1
           PReLU-353         [-1, 128, 64, 128]               1
           PReLU-354         [-1, 128, 64, 128]               1
RegularBottleneck-355         [-1, 128, 64, 128]               0
          Conv2d-356          [-1, 32, 64, 128]           4,096
     BatchNorm2d-357          [-1, 32, 64, 128]              64
           PReLU-358          [-1, 32, 64, 128]               1
           PReLU-359          [-1, 32, 64, 128]               1
           PReLU-360          [-1, 32, 64, 128]               1
           PReLU-361          [-1, 32, 64, 128]               1
          Conv2d-362          [-1, 32, 64, 128]           9,216
     BatchNorm2d-363          [-1, 32, 64, 128]              64
           PReLU-364          [-1, 32, 64, 128]               1
           PReLU-365          [-1, 32, 64, 128]               1
           PReLU-366          [-1, 32, 64, 128]               1
           PReLU-367          [-1, 32, 64, 128]               1
          Conv2d-368         [-1, 128, 64, 128]           4,096
     BatchNorm2d-369         [-1, 128, 64, 128]             256
           PReLU-370         [-1, 128, 64, 128]               1
           PReLU-371         [-1, 128, 64, 128]               1
           PReLU-372         [-1, 128, 64, 128]               1
           PReLU-373         [-1, 128, 64, 128]               1
       Dropout2d-374         [-1, 128, 64, 128]               0
           PReLU-375         [-1, 128, 64, 128]               1
           PReLU-376         [-1, 128, 64, 128]               1
           PReLU-377         [-1, 128, 64, 128]               1
           PReLU-378         [-1, 128, 64, 128]               1
RegularBottleneck-379         [-1, 128, 64, 128]               0
          Conv2d-380          [-1, 32, 64, 128]           4,096
     BatchNorm2d-381          [-1, 32, 64, 128]              64
           PReLU-382          [-1, 32, 64, 128]               1
           PReLU-383          [-1, 32, 64, 128]               1
           PReLU-384          [-1, 32, 64, 128]               1
           PReLU-385          [-1, 32, 64, 128]               1
          Conv2d-386          [-1, 32, 64, 128]           9,216
     BatchNorm2d-387          [-1, 32, 64, 128]              64
           PReLU-388          [-1, 32, 64, 128]               1
           PReLU-389          [-1, 32, 64, 128]               1
           PReLU-390          [-1, 32, 64, 128]               1
           PReLU-391          [-1, 32, 64, 128]               1
          Conv2d-392         [-1, 128, 64, 128]           4,096
     BatchNorm2d-393         [-1, 128, 64, 128]             256
           PReLU-394         [-1, 128, 64, 128]               1
           PReLU-395         [-1, 128, 64, 128]               1
           PReLU-396         [-1, 128, 64, 128]               1
           PReLU-397         [-1, 128, 64, 128]               1
       Dropout2d-398         [-1, 128, 64, 128]               0
           PReLU-399         [-1, 128, 64, 128]               1
           PReLU-400         [-1, 128, 64, 128]               1
           PReLU-401         [-1, 128, 64, 128]               1
           PReLU-402         [-1, 128, 64, 128]               1
RegularBottleneck-403         [-1, 128, 64, 128]               0
          Conv2d-404          [-1, 32, 64, 128]           4,096
     BatchNorm2d-405          [-1, 32, 64, 128]              64
           PReLU-406          [-1, 32, 64, 128]               1
           PReLU-407          [-1, 32, 64, 128]               1
           PReLU-408          [-1, 32, 64, 128]               1
           PReLU-409          [-1, 32, 64, 128]               1
          Conv2d-410          [-1, 32, 64, 128]           5,120
     BatchNorm2d-411          [-1, 32, 64, 128]              64
           PReLU-412          [-1, 32, 64, 128]               1
           PReLU-413          [-1, 32, 64, 128]               1
           PReLU-414          [-1, 32, 64, 128]               1
           PReLU-415          [-1, 32, 64, 128]               1
          Conv2d-416          [-1, 32, 64, 128]           5,120
     BatchNorm2d-417          [-1, 32, 64, 128]              64
           PReLU-418          [-1, 32, 64, 128]               1
           PReLU-419          [-1, 32, 64, 128]               1
           PReLU-420          [-1, 32, 64, 128]               1
           PReLU-421          [-1, 32, 64, 128]               1
          Conv2d-422         [-1, 128, 64, 128]           4,096
     BatchNorm2d-423         [-1, 128, 64, 128]             256
           PReLU-424         [-1, 128, 64, 128]               1
           PReLU-425         [-1, 128, 64, 128]               1
           PReLU-426         [-1, 128, 64, 128]               1
           PReLU-427         [-1, 128, 64, 128]               1
       Dropout2d-428         [-1, 128, 64, 128]               0
           PReLU-429         [-1, 128, 64, 128]               1
           PReLU-430         [-1, 128, 64, 128]               1
           PReLU-431         [-1, 128, 64, 128]               1
           PReLU-432         [-1, 128, 64, 128]               1
RegularBottleneck-433         [-1, 128, 64, 128]               0
          Conv2d-434          [-1, 32, 64, 128]           4,096
     BatchNorm2d-435          [-1, 32, 64, 128]              64
           PReLU-436          [-1, 32, 64, 128]               1
           PReLU-437          [-1, 32, 64, 128]               1
           PReLU-438          [-1, 32, 64, 128]               1
           PReLU-439          [-1, 32, 64, 128]               1
          Conv2d-440          [-1, 32, 64, 128]           9,216
     BatchNorm2d-441          [-1, 32, 64, 128]              64
           PReLU-442          [-1, 32, 64, 128]               1
           PReLU-443          [-1, 32, 64, 128]               1
           PReLU-444          [-1, 32, 64, 128]               1
           PReLU-445          [-1, 32, 64, 128]               1
          Conv2d-446         [-1, 128, 64, 128]           4,096
     BatchNorm2d-447         [-1, 128, 64, 128]             256
           PReLU-448         [-1, 128, 64, 128]               1
           PReLU-449         [-1, 128, 64, 128]               1
           PReLU-450         [-1, 128, 64, 128]               1
           PReLU-451         [-1, 128, 64, 128]               1
       Dropout2d-452         [-1, 128, 64, 128]               0
           PReLU-453         [-1, 128, 64, 128]               1
           PReLU-454         [-1, 128, 64, 128]               1
           PReLU-455         [-1, 128, 64, 128]               1
           PReLU-456         [-1, 128, 64, 128]               1
RegularBottleneck-457         [-1, 128, 64, 128]               0
          Conv2d-458          [-1, 32, 64, 128]           4,096
     BatchNorm2d-459          [-1, 32, 64, 128]              64
           PReLU-460          [-1, 32, 64, 128]               1
           PReLU-461          [-1, 32, 64, 128]               1
           PReLU-462          [-1, 32, 64, 128]               1
           PReLU-463          [-1, 32, 64, 128]               1
          Conv2d-464          [-1, 32, 64, 128]           9,216
     BatchNorm2d-465          [-1, 32, 64, 128]              64
           PReLU-466          [-1, 32, 64, 128]               1
           PReLU-467          [-1, 32, 64, 128]               1
           PReLU-468          [-1, 32, 64, 128]               1
           PReLU-469          [-1, 32, 64, 128]               1
          Conv2d-470         [-1, 128, 64, 128]           4,096
     BatchNorm2d-471         [-1, 128, 64, 128]             256
           PReLU-472         [-1, 128, 64, 128]               1
           PReLU-473         [-1, 128, 64, 128]               1
           PReLU-474         [-1, 128, 64, 128]               1
           PReLU-475         [-1, 128, 64, 128]               1
       Dropout2d-476         [-1, 128, 64, 128]               0
           PReLU-477         [-1, 128, 64, 128]               1
           PReLU-478         [-1, 128, 64, 128]               1
           PReLU-479         [-1, 128, 64, 128]               1
           PReLU-480         [-1, 128, 64, 128]               1
RegularBottleneck-481         [-1, 128, 64, 128]               0
          Conv2d-482          [-1, 32, 64, 128]           4,096
     BatchNorm2d-483          [-1, 32, 64, 128]              64
           PReLU-484          [-1, 32, 64, 128]               1
           PReLU-485          [-1, 32, 64, 128]               1
           PReLU-486          [-1, 32, 64, 128]               1
           PReLU-487          [-1, 32, 64, 128]               1
          Conv2d-488          [-1, 32, 64, 128]           9,216
     BatchNorm2d-489          [-1, 32, 64, 128]              64
           PReLU-490          [-1, 32, 64, 128]               1
           PReLU-491          [-1, 32, 64, 128]               1
           PReLU-492          [-1, 32, 64, 128]               1
           PReLU-493          [-1, 32, 64, 128]               1
          Conv2d-494         [-1, 128, 64, 128]           4,096
     BatchNorm2d-495         [-1, 128, 64, 128]             256
           PReLU-496         [-1, 128, 64, 128]               1
           PReLU-497         [-1, 128, 64, 128]               1
           PReLU-498         [-1, 128, 64, 128]               1
           PReLU-499         [-1, 128, 64, 128]               1
       Dropout2d-500         [-1, 128, 64, 128]               0
           PReLU-501         [-1, 128, 64, 128]               1
           PReLU-502         [-1, 128, 64, 128]               1
           PReLU-503         [-1, 128, 64, 128]               1
           PReLU-504         [-1, 128, 64, 128]               1
RegularBottleneck-505         [-1, 128, 64, 128]               0
          Conv2d-506          [-1, 32, 64, 128]           4,096
     BatchNorm2d-507          [-1, 32, 64, 128]              64
           PReLU-508          [-1, 32, 64, 128]               1
           PReLU-509          [-1, 32, 64, 128]               1
           PReLU-510          [-1, 32, 64, 128]               1
           PReLU-511          [-1, 32, 64, 128]               1
          Conv2d-512          [-1, 32, 64, 128]           5,120
     BatchNorm2d-513          [-1, 32, 64, 128]              64
           PReLU-514          [-1, 32, 64, 128]               1
           PReLU-515          [-1, 32, 64, 128]               1
           PReLU-516          [-1, 32, 64, 128]               1
           PReLU-517          [-1, 32, 64, 128]               1
          Conv2d-518          [-1, 32, 64, 128]           5,120
     BatchNorm2d-519          [-1, 32, 64, 128]              64
           PReLU-520          [-1, 32, 64, 128]               1
           PReLU-521          [-1, 32, 64, 128]               1
           PReLU-522          [-1, 32, 64, 128]               1
           PReLU-523          [-1, 32, 64, 128]               1
          Conv2d-524         [-1, 128, 64, 128]           4,096
     BatchNorm2d-525         [-1, 128, 64, 128]             256
           PReLU-526         [-1, 128, 64, 128]               1
           PReLU-527         [-1, 128, 64, 128]               1
           PReLU-528         [-1, 128, 64, 128]               1
           PReLU-529         [-1, 128, 64, 128]               1
       Dropout2d-530         [-1, 128, 64, 128]               0
           PReLU-531         [-1, 128, 64, 128]               1
           PReLU-532         [-1, 128, 64, 128]               1
           PReLU-533         [-1, 128, 64, 128]               1
           PReLU-534         [-1, 128, 64, 128]               1
RegularBottleneck-535         [-1, 128, 64, 128]               0
          Conv2d-536          [-1, 32, 64, 128]           4,096
     BatchNorm2d-537          [-1, 32, 64, 128]              64
           PReLU-538          [-1, 32, 64, 128]               1
           PReLU-539          [-1, 32, 64, 128]               1
           PReLU-540          [-1, 32, 64, 128]               1
           PReLU-541          [-1, 32, 64, 128]               1
          Conv2d-542          [-1, 32, 64, 128]           9,216
     BatchNorm2d-543          [-1, 32, 64, 128]              64
           PReLU-544          [-1, 32, 64, 128]               1
           PReLU-545          [-1, 32, 64, 128]               1
           PReLU-546          [-1, 32, 64, 128]               1
           PReLU-547          [-1, 32, 64, 128]               1
          Conv2d-548         [-1, 128, 64, 128]           4,096
     BatchNorm2d-549         [-1, 128, 64, 128]             256
           PReLU-550         [-1, 128, 64, 128]               1
           PReLU-551         [-1, 128, 64, 128]               1
           PReLU-552         [-1, 128, 64, 128]               1
           PReLU-553         [-1, 128, 64, 128]               1
       Dropout2d-554         [-1, 128, 64, 128]               0
           PReLU-555         [-1, 128, 64, 128]               1
           PReLU-556         [-1, 128, 64, 128]               1
           PReLU-557         [-1, 128, 64, 128]               1
           PReLU-558         [-1, 128, 64, 128]               1
RegularBottleneck-559         [-1, 128, 64, 128]               0
          Conv2d-560          [-1, 64, 64, 128]           8,192
     BatchNorm2d-561          [-1, 64, 64, 128]             128
     MaxUnpool2d-562         [-1, 64, 128, 256]               0
          Conv2d-563          [-1, 32, 64, 128]           4,096
     BatchNorm2d-564          [-1, 32, 64, 128]              64
            ReLU-565          [-1, 32, 64, 128]               0
            ReLU-566          [-1, 32, 64, 128]               0
            ReLU-567          [-1, 32, 64, 128]               0
            ReLU-568          [-1, 32, 64, 128]               0
 ConvTranspose2d-569         [-1, 32, 128, 256]           9,216
     BatchNorm2d-570         [-1, 32, 128, 256]              64
            ReLU-571         [-1, 32, 128, 256]               0
            ReLU-572         [-1, 32, 128, 256]               0
            ReLU-573         [-1, 32, 128, 256]               0
            ReLU-574         [-1, 32, 128, 256]               0
          Conv2d-575         [-1, 64, 128, 256]           2,048
     BatchNorm2d-576         [-1, 64, 128, 256]             128
            ReLU-577         [-1, 64, 128, 256]               0
            ReLU-578         [-1, 64, 128, 256]               0
            ReLU-579         [-1, 64, 128, 256]               0
            ReLU-580         [-1, 64, 128, 256]               0
       Dropout2d-581         [-1, 64, 128, 256]               0
            ReLU-582         [-1, 64, 128, 256]               0
            ReLU-583         [-1, 64, 128, 256]               0
            ReLU-584         [-1, 64, 128, 256]               0
            ReLU-585         [-1, 64, 128, 256]               0
UpsamplingBottleneck-586         [-1, 64, 128, 256]               0
          Conv2d-587         [-1, 16, 128, 256]           1,024
     BatchNorm2d-588         [-1, 16, 128, 256]              32
            ReLU-589         [-1, 16, 128, 256]               0
            ReLU-590         [-1, 16, 128, 256]               0
            ReLU-591         [-1, 16, 128, 256]               0
            ReLU-592         [-1, 16, 128, 256]               0
          Conv2d-593         [-1, 16, 128, 256]           2,304
     BatchNorm2d-594         [-1, 16, 128, 256]              32
            ReLU-595         [-1, 16, 128, 256]               0
            ReLU-596         [-1, 16, 128, 256]               0
            ReLU-597         [-1, 16, 128, 256]               0
            ReLU-598         [-1, 16, 128, 256]               0
          Conv2d-599         [-1, 64, 128, 256]           1,024
     BatchNorm2d-600         [-1, 64, 128, 256]             128
            ReLU-601         [-1, 64, 128, 256]               0
            ReLU-602         [-1, 64, 128, 256]               0
            ReLU-603         [-1, 64, 128, 256]               0
            ReLU-604         [-1, 64, 128, 256]               0
       Dropout2d-605         [-1, 64, 128, 256]               0
            ReLU-606         [-1, 64, 128, 256]               0
            ReLU-607         [-1, 64, 128, 256]               0
            ReLU-608         [-1, 64, 128, 256]               0
            ReLU-609         [-1, 64, 128, 256]               0
RegularBottleneck-610         [-1, 64, 128, 256]               0
          Conv2d-611         [-1, 16, 128, 256]           1,024
     BatchNorm2d-612         [-1, 16, 128, 256]              32
            ReLU-613         [-1, 16, 128, 256]               0
            ReLU-614         [-1, 16, 128, 256]               0
            ReLU-615         [-1, 16, 128, 256]               0
            ReLU-616         [-1, 16, 128, 256]               0
          Conv2d-617         [-1, 16, 128, 256]           2,304
     BatchNorm2d-618         [-1, 16, 128, 256]              32
            ReLU-619         [-1, 16, 128, 256]               0
            ReLU-620         [-1, 16, 128, 256]               0
            ReLU-621         [-1, 16, 128, 256]               0
            ReLU-622         [-1, 16, 128, 256]               0
          Conv2d-623         [-1, 64, 128, 256]           1,024
     BatchNorm2d-624         [-1, 64, 128, 256]             128
            ReLU-625         [-1, 64, 128, 256]               0
            ReLU-626         [-1, 64, 128, 256]               0
            ReLU-627         [-1, 64, 128, 256]               0
            ReLU-628         [-1, 64, 128, 256]               0
       Dropout2d-629         [-1, 64, 128, 256]               0
            ReLU-630         [-1, 64, 128, 256]               0
            ReLU-631         [-1, 64, 128, 256]               0
            ReLU-632         [-1, 64, 128, 256]               0
            ReLU-633         [-1, 64, 128, 256]               0
RegularBottleneck-634         [-1, 64, 128, 256]               0
          Conv2d-635         [-1, 16, 128, 256]           1,024
     BatchNorm2d-636         [-1, 16, 128, 256]              32
     MaxUnpool2d-637         [-1, 16, 256, 512]               0
          Conv2d-638         [-1, 16, 128, 256]           1,024
     BatchNorm2d-639         [-1, 16, 128, 256]              32
            ReLU-640         [-1, 16, 128, 256]               0
            ReLU-641         [-1, 16, 128, 256]               0
            ReLU-642         [-1, 16, 128, 256]               0
            ReLU-643         [-1, 16, 128, 256]               0
 ConvTranspose2d-644         [-1, 16, 256, 512]           2,304
     BatchNorm2d-645         [-1, 16, 256, 512]              32
            ReLU-646         [-1, 16, 256, 512]               0
            ReLU-647         [-1, 16, 256, 512]               0
            ReLU-648         [-1, 16, 256, 512]               0
            ReLU-649         [-1, 16, 256, 512]               0
          Conv2d-650         [-1, 16, 256, 512]             256
     BatchNorm2d-651         [-1, 16, 256, 512]              32
            ReLU-652         [-1, 16, 256, 512]               0
            ReLU-653         [-1, 16, 256, 512]               0
            ReLU-654         [-1, 16, 256, 512]               0
            ReLU-655         [-1, 16, 256, 512]               0
       Dropout2d-656         [-1, 16, 256, 512]               0
            ReLU-657         [-1, 16, 256, 512]               0
            ReLU-658         [-1, 16, 256, 512]               0
            ReLU-659         [-1, 16, 256, 512]               0
            ReLU-660         [-1, 16, 256, 512]               0
UpsamplingBottleneck-661         [-1, 16, 256, 512]               0
          Conv2d-662          [-1, 4, 256, 512]              64
     BatchNorm2d-663          [-1, 4, 256, 512]               8
            ReLU-664          [-1, 4, 256, 512]               0
            ReLU-665          [-1, 4, 256, 512]               0
            ReLU-666          [-1, 4, 256, 512]               0
            ReLU-667          [-1, 4, 256, 512]               0
          Conv2d-668          [-1, 4, 256, 512]             144
     BatchNorm2d-669          [-1, 4, 256, 512]               8
            ReLU-670          [-1, 4, 256, 512]               0
            ReLU-671          [-1, 4, 256, 512]               0
            ReLU-672          [-1, 4, 256, 512]               0
            ReLU-673          [-1, 4, 256, 512]               0
          Conv2d-674         [-1, 16, 256, 512]              64
     BatchNorm2d-675         [-1, 16, 256, 512]              32
            ReLU-676         [-1, 16, 256, 512]               0
            ReLU-677         [-1, 16, 256, 512]               0
            ReLU-678         [-1, 16, 256, 512]               0
            ReLU-679         [-1, 16, 256, 512]               0
       Dropout2d-680         [-1, 16, 256, 512]               0
            ReLU-681         [-1, 16, 256, 512]               0
            ReLU-682         [-1, 16, 256, 512]               0
            ReLU-683         [-1, 16, 256, 512]               0
            ReLU-684         [-1, 16, 256, 512]               0
RegularBottleneck-685         [-1, 16, 256, 512]               0
 ConvTranspose2d-686        [-1, 19, 512, 1024]           2,736
================================================================
Total params: 358,336
Trainable params: 358,336
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 6.00
Forward/backward pass size (MB): 4688.00
Params size (MB): 1.37
Estimated Total Size (MB): 4695.37
----------------------------------------------------------------
'''
# summary(model,(3,512,1024))
# ================================================================
# Total params: 358,336
# Trainable params: 358,336
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 6.00
# Forward/backward pass size (MB): 4688.00
# Params size (MB): 1.37
# Estimated Total Size (MB): 4695.37
# ----------------------------------------------------------------

# summary(model,(3,1024,2048))
# ================================================================
# Total params: 358,336
# Trainable params: 358,336
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 24.00
# Forward/backward pass size (MB): 14016.00
# Params size (MB): 1.37
# Estimated Total Size (MB): 14041.37
# ----------------------------------------------------------------
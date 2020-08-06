# *- coding: utf-8 -*
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import collections

import pandas as pd
import time
from collections import OrderedDict
import numpy as np

# from torchsummary import summary
from tools.flops_counter.ptflops import get_model_complexity_info

from model.ESPNet_v2.SegmentationModel import EESPNet_Seg
from model.CGNet import CGNet
from model.ContextNet import ContextNet
from model.DABNet import DABNet
from model.EDANet import EDANet
from model.ENet import ENet
from model.ERFNet import ERFNet
from model.ESNet import ESNet
from model.ESPNet import ESPNet
from model.FastSCNN import FastSCNN
from model.FPENet import FPENet
from model.FSSNet import FSSNet
from model.LEDNet import LEDNet
from model.LinkNet import LinkNet
from model.SegNet import SegNet
from model.SQNet import SQNet
from model.UNet import UNet

models = {
    # 'EESPNet_Seg': EESPNet_Seg,
    # 'CGNet': CGNet,
    # 'ContextNet':ContextNet,
    # 'DABNet':DABNet,
    # 'EDANet': EDANet,
    # 'ENet': ENet,
    # 'ERFNet': ERFNet,
    # 'ESNet': ESNet,
    # 'ESPNet': ESPNet,
    # 'FastSCNN': FastSCNN,
    'FPENet': FPENet,
    # 'FSSNet':FSSNet,
    # 'LEDNet': LEDNet,
    # 'LinkNet':LinkNet,
    # 'SegNet':SegNet,
    # 'SQNet':SQNet,
    # 'UNet':UNet,
}

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
    return total_params, trainable_params, total_input_size, total_output_size, total_params_size, total_size

file_path = './net_summary.csv'


if __name__ == '__main__':

    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns = ['net','size','Flops(GMAC)','params(M)','params_A','params_T','Input(MB)','F/B (MB)','Params(MB)','Total(MB)'])
        df.to_csv(file_path, mode='w', header=True, index=False)

    # models = collections.OrderedDict()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for i in [1,2,4,6,8,10,12]:
    for i in [16]:
        img_h,img_w =32*i,64*i
        for name, network in models.items():
            print(name, network)
            model = network(classes=11).to(device)

            total_params, trainable_params, total_input_size, total_output_size, total_params_size, total_size = summary(
                model, (3, img_h, img_w))

            # time.sleep(1)

            flops, params = get_model_complexity_info(model, (3, img_h, img_w),
                                                      as_strings=False,
                                                      print_per_layer_stat=True,
                                                      ost=sys.stdout)
            # print('Flops: ' + flops)
            # print('Params: ' + params)
            result = dict()
            result['net'] = name
            result['size'] = '{}x{}'.format(img_h,img_w)
            result['Flops'] = flops/(10**9)
            result['params'] = params/(10**6)
            result['params_A'] =total_params.item()
            result['params_T'] = trainable_params.item()
            result['Input(MB)'] = total_input_size
            result['F/B (MB)'] = total_output_size
            result['Params(MB)'] = total_params_size
            result['Total(MB)'] = total_size
            # results.append(result)
            df = pd.DataFrame([result])
            df.to_csv(file_path, mode = 'a', header=False, index=False)

    results_df = pd.read_csv(file_path).drop_duplicates()
    print(results_df)
    results_df.to_csv(file_path, mode='w', header=True, index=False)



    # import pandas as pd
    # import numpy as np
    #
    # df = pd.DataFrame(np.random.randn(5, 3),index=['a', 'c', 'e', 'f', 'h'],columns = ['one', 'two', 'three'])  # 创建一个数据表格
    # df['four'] = 'bar'  # 加入新的一列
    # df['five'] = df['one'] > 0  # 加入新的一列，通过判断数据大小加入bool型列
    # df['six'] = 'ball'
    #
    # print(df)

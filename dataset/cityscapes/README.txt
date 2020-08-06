# README
## 关于超神经 Hyper.AI
超神经 Hyper.AI（https://hyper.ai）是科技实验媒体，专注报道人工智能与其适用场景。致力于推动中文领域对机器智能的认知与普及，探讨机器智能的对社会的影响。超神经为提高科研效率，提供大陆范围内最快最全的公开数据集下载节点、人工智能百科词条等多个产品，服务产业相关从业者和科研院所的师生。

## 关于数据集
- 数据集名称：Cityscapes 立体视频数据集
- 发布机构：Daimler AG（戴姆勒股份公司）, Darmstadt University of Technology（达姆施塔特工业大学）, Max Planck Institute
- 发布地址：https://www.cityscapes-dataset.com/
- 下载地址：https://hyper.ai/datasets/5205
- 简介：Cityscapes 数据集是包含来自 50 个不同城市的街景中记录的各种立体视频序列的一个新的大型数据集拥有更大的 20000 个弱注释帧和 5000 帧的高质量像素级注释。该数据集专注于对城市街景的语义理解，旨在用于评估视觉算法在语义城市场景理解的两个主要任务中的表现：像素级和实例级语义标签; 支持旨在利用大量（弱）注释数据的研究，例如用于训练深度神经网络。

Cityscapes 数据集于 2016 年由戴姆勒股份公司，马克斯普朗克信息学研究所，达姆施塔特工业大学视觉推理实验室等中的人员组成的 Cityscapes 团队发布，相关论文为 The Cityscapes Dataset for Semantic Urban Scene Understanding。


1.download  scripts 
	https://github.com/mcordts/cityscapesScripts
2.download dataset 可通过scripts下载，或手动下载，unzip 
   存放结构：
   cityscapes/
	cityscapesscripts
	gtFine
                leftImg8bit
	...     
3.preparation 预处理
    createTrainIdLabelImgs.py (生成19类的语义分割label)
4.annotation 提供了手工标注的GUI界面
5.viewer        提供了浏览标注的GUI界面
6.evaluation 提供了标准化的test result的接口
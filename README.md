# FishNet

This repo holds the Tensorflow implementation of the paper:

[FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)
, Shuyang Sun, Jiangmiao Pang, Jianping Shi, Shuai Yi, Wanli Ouyang, NeurIPS 2018.

The original implmentation code(PyTorch) : https://github.com/kevin-ssy/FishNet

### Prerequisites
- Python 2.7.15
- Tensorflow 1.12
- Tensorflow models(official module) : https://github.com/tensorflow/models

### Training
1. You have to download the ImageNet dataset and convert it to TFRecord format. Please refer to [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy)

2. You can train ImageNet dataset by the following command
```
python imagenet_main.py
```

3. You can get additional information for the parameters by the following command
```
python imagenet_main.py -h
```
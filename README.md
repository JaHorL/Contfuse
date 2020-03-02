# Contfuse: Deep Continuous Fusion for Multi-Sensor 3D Object Detection
## Introduction

It's a unofficial tensorflow Implementation of Contfuse: Deep Continuous Fusion for Multi-Sensor 3D Object Detection. It uses C++ \  CUDA C \ Python to complete this project.

## Train on KITTI Dataset

I split KITTI train data to testing data \  training  data \   verification data.

```shell
kitti dataset:   <-- 7481 train data
|-- data_object_calib   <-- 7481
    |--calib
|-- image_2    <-- 7481
|-- lidar_files   <-- 7481
|-- testing
    |-- label_files <-- 1000
|-- training
    |-- label_files <-- 6431
|-- val
    |-- label_files <-- 50
```

## How to use it?

### Dependencies

tensorflow 1.14
numpy 1.16
opencv 3.4
easydict
cudnn 7.6.0
cuda 10.1
python 3.7
tqdm

### train

step. 1  Generate dataset index document, you need modified dataset path.

```shell
cd src/scripts
python gen_dataset_idx.py
```

step. 2  train

```shell
python train.py
```

### predict

```
python predict.py
```

## Credit

CONTFUSE: Deep Continuous Fusion for Multi-Sensor 3D Object Detection

PIXOR: Real-time 3D Object Detection from Point Clouds
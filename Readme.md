# SpineNet-Pytorch
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinenet-learning-scale-permuted-backbone-for/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=spinenet-learning-scale-permuted-backbone-for)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinenet-learning-scale-permuted-backbone-for/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=spinenet-learning-scale-permuted-backbone-for)<br>

![demo image](demo/coco_test_12510.jpg)

[SpineNet](https://arxiv.org/abs/1912.05027) is a scale-permuted backbone for object detection, proposed by Google Brain in CVPR 2020. This project is a kind of implementation of SpineNet using mmdetection.

It is highly based on the
* [yan-roo/SpineNet-Pytorch](https://github.com/yan-roo/SpineNet-Pytorch)
* [lucifer443/SpineNet-Pytorch](https://github.com/lucifer443/SpineNet-Pytorch)
* The paper [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027)
* [Official TensorFlow Implementation](https://github.com/tensorflow/tpu/tree/master/models/official/detection)

## Models
### Instance Segmentation Baselines
#### Mask R-CNN (Trained from scratch)
| Backbone     | Resolution  |box AP|mask AP| Params | FLOPs   |box mAP <br> (paper)|mask mAP <br> (paper)| Params <br> (paper) | FLOPs <br> (paper) | Download |
| ------------ | ----------  | ---- | ----- | ------ | ------- | ------------------ | ------------------- | ------------------- | ------------------ | -------- |
| [SpineNet-49S](configs/spinenet/mask_rcnn_spinenet_49S_B_8gpu_640.py)|   640x640   | 39.7 | 34.9 | 13.92M | 63.77B   | 39.3 | 34.8 | 13.9M | 60.2B  | [model](https://drive.google.com/file/d/1WEa7y8kFXPoCtDEeNpTzJrKpVlMbjiGG/view?usp=sharing) |

## Installation

### 1. Install mmdetection

   This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.1.0+8732ed9).
   
   Please refer to [INSTALL.md](docs/INSTALL.md) for more information.

   a. Create a conda virtual environment and activate it.
   ```shell
   conda create -n mmlab python=3.7 -y
   conda activate mmlab
   ```

   b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

   ```shell
   conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
   ```
   c. Install mmcv
   
   ```shell
   pip install mmcv==0.4.3
   ```  
   
   d. Clone the mmdetection repository.

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   git checkout 8732ed9
   ```

   e. Install build requirements and then install mmdetection.
   (We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)

   ```shell
   pip install -r requirements/build.txt
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   pip install -v -e .  # or "python setup.py develop"
   ```

## Getting started
### 1. Copy the codes to mmdetection directory

```shell
git clone https://github.com/yan-roo/SpineNet-Pytorch.git
cp -r mmdet/ mmdetection/
cp -r configs/ mmdetection/
```

### 2. Prepare dataset (COCO)

```shell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip ${train/val/test}2017.zip
mv ${train/val/test}2017  ${mmdetection/data/coco/}
```

  The directories should be arranged like this:

     >   mmdetection
     >     ├── mmdet
     >     ├── tools
     >     ├── configs
     >     ├── data
     >     │   ├── coco
     >     │   │   ├── annotations
     >     │   │   ├── train2017
     >     │   │   ├── val2017
     >     │   │   ├── test2017


### 3. Train a model

**\*Important\***: The default learning rate in SpineNet-49S config files is for 8 GPUs and [32 img/gpu](https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/spinenet_49S_B_8gpu.py#L87) (batch size = 8*32 = 256).

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU.

e.g., lr=0.28 for 8 GPUs * 32 img/gpu and lr=0.07 for 8 GPUs * 8 img/gpu.

You also can set the [warm-up iterations](https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/spinenet_49S_B_8gpu.py#L117).

e.g., warmup_iters=2000 for batch size 256 and warmup_iters=8000 for batch size 64

#### Train with a single GPU
Modify [config/spinenet/model.py Line5](https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/spinenet_49S_B_8gpu.py#L5)
type='SyncBN' -> type='BN'

```shell
CONFIG_FILE=configs/spinenet/spinenet_49S_B_8gpu.py
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

_[optional arguments]:_ --resume_from ${epoch_.pth} / --validate

#### Train with multiple GPUs

```shell
CONFIG_FILE=configs/spinenet/spinenet_49S_B_8gpu.py
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
### 4. Calculate parameters and FLOPs

```shell
python tools/get_flops.py ${CONFIG_FILE} --shape $SIZE $SIZE
```

### 5. Evaluation

   ```shell
   python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out  ${OUTPUT_FILE} --eval bbox
   python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out  ${OUTPUT_FILE} --eval bbox segm
   ```

More usages can reference [GETTING_STARTED.md](docs/GETTING_STARTED.md) or [MMDetection documentation](https://mmdetection.readthedocs.io/).

## Issues & FAQ

   1. ModuleNotFoundError: No module named 'mmcv.cnn.weight_init'

      ```
      pip install mmcv==0.4.3	
      ```

   2. [ImportError: libtorch_cpu.so: cannot open shared object file: No such file or directory](https://github.com/open-mmlab/mmdetection/issues/2627)

      ```
      rm -r build
      python setup.py develop
      ```
   3. AssertionError: Default process group is not initialized
   
      Modify [config/spinenet/model.py Line5](https://github.com/yan-roo/SpineNet-Pytorch/blob/master/configs/spinenet/spinenet_49S_B_8gpu.py#L5) type='SyncBN' -> type='BN'

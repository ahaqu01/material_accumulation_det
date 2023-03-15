# material_accumulation_det API

## 1、接口说明

***material_accumulation_det模型仍在优化中，会持续更新...***

该接口为智慧城管相关的物品检测算法。

目前模型为yolox-s，基于datasetv1.1，支持检测的类型如下：

| class_name (ch)                | class_name (eh)          |
| :----------------------------- | ------------------------ |
| 箱盒状的物品                   | box                      |
| 塑料袋装着、塑料膜覆盖着的物品 | plastic                  |
| 建材、工具                     | tools                    |
| 盆、碗、罐、桶类形状的物品     | pot                      |
| 水果、蔬菜、干货               | fruit_vegetable_drygoods |
| 肉摊                           | meat                     |
| 植物花草摊                     | plant                    |
| 服装、鞋子                     | clothes                  |
| 其他                           | others                   |
| 柜状物                         | cabinet                  |
| 桌、椅、凳                     | tables_chairs            |
| 街边食品车                     | street_food_truck        |
| 篷布、大伞                     | Tarpaulin                |

目前该接口支持pytorch模型和TensorRT加速模型的推理

加速前后模型速度对比见语雀：https://b2i2t.yuque.com/docs/share/27104cdc-0e60-4e0c-9637-9fb4ee8c9ceb?# 《YOLOX-S模型TensorRT加速》

模型检测结果可视化样例如下图：

<img src="./vis_o_0000_0018.jpg" alt="vis_o_0000_0018" style="zoom:50%;" />

## 2、依赖配置

首先用pip install -r requirements.txt安装其他依赖

然后安装tensorrt==8.0.0.3和torch2trt，安装方法见语雀: https://b2i2t.yuque.com/docs/share/fab72f58-09c8-4389-a427-bf83380dd006?# 《TensorRT/torch2trt安装》

## 3、Useage

#### 3.1. used as source code

###### step1. 依赖配置，详情见二

###### step2. 参数配置

通过修改material_accumulation_detection/configs/config.py文件进行参数配置

其中test_cfg中的trt可以用来控制是否开启TRT加速

```python
model = dict(
    train_class="0,1,2,3,4,5,6,7,8,9,10,11,12",
    num_classes=13,
    # factor of model depth/width
    depth=0.33,
    width=0.50,
    # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
    act="silu",
)

test_cfg = dict(
    input_size=(640, 640),# inference图像尺寸
    conf=0.2,# boxes whose scores are less than test_conf will be filtered
    nmsthre=0.65,# nms threshold
    fuse=False,
    fp16=False,
    legacy=False,
    gpu_id="0", # GPU设备ID
    trt=True, # 是否开启TensorRT加速
    trt_workspace=32,
)
```

###### step3. use it as submodule

在自己的项目目录下，git submodule add https://gitlab.ictidei.cn/band-intel-center/Algorithm-platform/material_accumulation_det.git

便会在项目目录下下载到material_accumulation_det相关代码

下载完成后，便可在自己项目中使用material_accumulation_det API，**使用样例**如下：

```python
from material_accumulation_det.material_accumulation_detection import MadetPredictor
import cv2
import os

madet_pred = MadetPredictor(
    model_cfg="/workspace/material_accumulation_det/material_accumulation_detection/configs/config.py", # 配置文件路径
    model_weights="/workspace/material_accumulation_det/material_accumulation_detection/weights/epoch_86_ckpt.pth", # 权重路径
    label_map_path="/workspace/material_accumulation_det/material_accumulation_detection/configs/label_map.py", # label map文件路径
)

filepath = "/workspace/material_accumulation/YOLOX/datasets/smart_urban_management/image/o_0000_0018.jpg"
img = cv2.imread(filepath)

# test one image
pred = madet_pred.single_inference(img)
# 可视化
vis_res = madet_pred.visual(img, pred, cls_conf=0.35)
out_path = os.path.join("./", "vis_" + filepath.split("/")[-1])
cv2.imwrite(out_path, vis_res)
```

#### 3.2. used by pip package

###### step1. 依赖配置，详情见二

###### step2. pip安装material-accumulation-detection包

pip install material-accumulation-detection --extra-index-url https://<personal_access_token_name>:<personal_access_token>@gitlab.ictidei.cn/api/v4/projects/114/packages/pypi/simple --force-reinstall

###### step3. 使用样例

```python
import cv2
import os
from material_accumulation_detection import MadetPredictor
filepath = "/workspace/material_accumulation/YOLOX/datasets/smart_urban_management/image/o_0000_0018.jpg"
img = cv2.imread(filepath)

madet_pred = MadetPredictor(
    model_cfg="/workspace/material_accumulation_det/material_accumulation_detection/configs/config.py", # 配置文件路径
    model_weights="/workspace/material_accumulation_det/material_accumulation_detection/weights/epoch_86_ckpt.pth", # 权重路径
    label_map_path="/workspace/material_accumulation_det/material_accumulation_detection/configs/label_map.py", # label map文件路径
)
pred = madet_pred.single_inference(img)
vis_res = madet_pred.visual(img, pred, cls_conf=0.35)
out_path = os.path.join("./", "vis_" + filepath.split("/")[-1])
cv2.imwrite(out_path, vis_res)
```


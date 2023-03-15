# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import time
from loguru import logger

import cv2

import torch
import tensorrt as trt
from torch2trt import torch2trt, TRTModule

import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(SCRIPT_DIR)
from utils.data_augment import ValTransform
from utils.boxes import postprocess
from utils.visualize import vis
from utils.get_cfg import get_cfgs
from models.yolox import YOLOX
from models.yolo_head import YOLOXHead
from models.yolo_pafpn import YOLOPAFPN
from models.model_utils import fuse_model


class MadetPredictor(object):

    def __init__(
            self,
            model_cfg="./configs/config.py",
            model_weights="./weights/epoch_86_ckpt.pth",
            label_map_path="./configs/label_map.py"
    ):
        # load cfgs
        self.cfgs = get_cfgs(model_cfg)
        logger.info("loaded configs done.")

        # load label_map
        cls_names = get_cfgs(label_map_path).MATERIAL_ACCUMULATION_CLASSES_DICT
        self.cls_names = [item[1] for item in sorted(cls_names.items(), key=lambda x: x[0], reverse=False)]

        # set envs
        gpu_id = self.cfgs.test_cfg["gpu_id"]
        self.device = torch.device('cuda:{}'.format(str(gpu_id)) if torch.cuda.is_available() else 'cpu')

        # create model
        self.depth = self.cfgs.model["depth"]
        self.width = self.cfgs.model["width"]
        self.act = self.cfgs.model["act"]
        self.num_classes = self.cfgs.model["num_classes"]
        self.model = self.get_model()
        logger.info("create model done.")

        # load ckpt
        ckpt = torch.load(model_weights, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        # set eval mode
        self.model.eval()

        # to gpu device
        self.model.to(self.device)

        # fuse
        if self.cfgs.test_cfg["fuse"]:
            self.model = fuse_model(self.model)
            logger.info("model fused")

        self.input_size = self.cfgs.test_cfg["input_size"]

        # if trt, create trt engine
        if self.cfgs.test_cfg["trt"]:
            self.model.head.decode_in_inference = False
            self.decoder = self.model.head.decode_outputs
            engine_file = os.path.join(os.path.abspath(os.path.join(model_weights, "..")), "model_trt.engine")
            pth_file = os.path.join(os.path.abspath(os.path.join(model_weights, "..")), "model_trt.pth")
            x = torch.ones(1, 3, self.input_size[0], self.input_size[1]).to(self.device)
            if not os.path.exists(pth_file):
                model_trt = torch2trt(
                    self.model,
                    [x],
                    fp16_mode=True,
                    log_level=trt.Logger.INFO,
                    max_workspace_size=(1 << self.cfgs.test_cfg["trt_workspace"]),
                    max_batch_size=1
                )
                torch.save(model_trt.state_dict(), pth_file)
                logger.info("Converted TensorRT model engine file is saved for python inference.")
                if not os.path.exists(engine_file):
                    with open(engine_file, "wb") as f:
                        f.write(model_trt.engine.serialize())
                    logger.info("Converted TensorRT model engine file is saved for c++ inference.")
            model_trt = TRTModule()
            pth_weights = torch.load(pth_file)
            model_trt.load_state_dict(pth_weights)
            self.model(x)
            self.model = model_trt
            logger.info("TensorRT model engine loaded")
        else:
            self.decoder = None
            # to FP16
            if self.cfgs.test_cfg["fp16"]:
                self.model.half()

        # data augment
        self.preproc = ValTransform(legacy=self.cfgs.test_cfg["legacy"])

        # test thrs
        self.confthre = self.cfgs.test_cfg["conf"]
        self.nmsthre = self.cfgs.test_cfg["nmsthre"]

    def get_model(self):
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        model = YOLOX(backbone, head)
        return model

    @torch.no_grad()
    def single_inference(self, img):
        # img, ndarray, shape=(h, w, 3), channel = BRR
        img = img.copy()
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.input_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)
        if not self.cfgs.test_cfg["trt"] and self.cfgs.test_cfg["fp16"]:
            img = img.half()  # to FP16

        t0 = time.time()
        outputs = self.model(img)
        if self.decoder is not None:
            outputs = self.decoder(outputs, dtype=outputs.type())
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True,
        )
        output = outputs[0].cpu()
        logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        if output is None:
            return None
        else:
            res = torch.zeros(size=(output.shape[0], 6))
            res[:, 0:4] = output[:, 0:4] / ratio
            res[:, 4] = output[:, 4] * output[:, 5]
            res[:, 5] = output[:, 6]
            return res

    def visual(self, img, output, cls_conf=0.35):
        bboxes = output[:, 0:4]
        cls = output[:, 5]
        scores = output[:, 4]
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


if __name__ == "__main__":
    madet_pred = MadetPredictor()
    filepath = "/workspace/material_accumulation/YOLOX/datasets/smart_urban_management/image/o_0000_0018.jpg"
    img = cv2.imread(filepath)

    # test one image
    pred = madet_pred.single_inference(img)
    vis_res = madet_pred.visual(img, pred, cls_conf=0.35)
    out_path = os.path.join("./", "vis_" + filepath.split("/")[-1])
    cv2.imwrite(out_path, vis_res)

    # speed test
    # h, w = img.shape[:2]
    # print("image size wxh: {}x{}".format(w, h))
    # s_t = time.time()
    # for i in range(1000):
    #     pred = madet_pred.single_inference(img)
    # e_t = time.time()
    # print(pred)
    # logger.info("average infer time: {:.4f}s".format((e_t - s_t) / 1000))
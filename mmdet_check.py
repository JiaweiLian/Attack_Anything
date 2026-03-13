import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
import os

model_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint = '../mmdetection-master/models/faster_rcnn_r50_fpn_2x_coco.pth'
model = init_detector(model_cfg, checkpoint, device='cuda:0')

print(model.cfg.data.test.pipeline)

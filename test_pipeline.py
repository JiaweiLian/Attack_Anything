import torch
import mmcv
from mmdet.apis import init_detector
import numpy as np

model_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint = '../mmdetection-master/models/faster_rcnn_r50_fpn_2x_coco.pth'
model = init_detector(model_cfg, checkpoint, device='cuda:0')

import sys
sys.path.append('.')
from utils import InferenceDetector_mmdet

img_tensor = torch.rand(1, 3, 416, 416).cuda()
img_cpu = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0

InferenceDetector = InferenceDetector_mmdet('configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py')
data = InferenceDetector(model, img_cpu)
print(data.keys())
print("img meta:", data['img_metas'][0].data)
print("img tensor shape:", data['img'][0].shape)
print("img tensor mean:", data['img'][0].mean())

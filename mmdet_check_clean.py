import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from mmdet.apis import init_detector
from mmdet.models.detectors.base import BaseDetector
from tqdm import tqdm

model_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
checkpoint = '../mmdetection-master/models/faster_rcnn_r50_fpn_2x_coco.pth'
model = init_detector(model_cfg, checkpoint, device='cuda:0')
model.eval()

ann_file = '../Datasets/coco/annotations/instances_val2017.json'
img_prefix = '../Datasets/coco/val2017/'
cocoGt = COCO(ann_file)
img_ids = cocoGt.getImgIds()[:50] 
results = []
config_img_size = 416

for idx in tqdm(img_ids):
    img_info = cocoGt.loadImgs(idx)[0]
    img_path = os.path.join(img_prefix, img_info['file_name'])
    
    # 1. Load Image
    image_pil = Image.open(img_path).convert('RGB')
    w, h = image_pil.size
    
    if w == h:
        padded_img = image_pil
        x_offset, y_offset = 0, 0
        pad_scale = config_img_size / w
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(image_pil, (int(padding), 0))
            x_offset, y_offset = int(padding), 0
            pad_scale = config_img_size / h
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(image_pil, (0, int(padding)))
            x_offset, y_offset = 0, int(padding)
            pad_scale = config_img_size / w

    padded_img = padded_img.resize((config_img_size, config_img_size))
    img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        img_cpu = img_tensor[0].detach().cpu().numpy()
        if img_cpu.shape[0] == 3:
            img_cpu = img_cpu.transpose(1, 2, 0)
        img_cpu = img_cpu * 255.0
        
        # We need InferenceDetector logic directly:
        from mmdet.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        
        # MMDetection inference code internally does data processing and resizing. 
        # By bypassing its pipeline and forcing rescale=False we got 0.31 map. 
        # Natively it's 0.59 mAP! 
        # Let's inspect the discrepancy!

import patch_config
from train import PatchTrainer
from evaluate import get_mmdet_predictions
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np

config = patch_config.patch_configs['faster_rcnn']()
trainer = PatchTrainer('faster_rcnn')
model = trainer.model.eval()

from pycocotools.coco import COCO
cocoGt = COCO('../Datasets/coco/annotations/instances_val2017.json')
img_info = cocoGt.loadImgs(list(cocoGt.imgs.keys())[0])[0]
img_path = '../Datasets/coco/val2017/' + img_info['file_name']
orig_img = Image.open(img_path).convert('RGB')

img_tensor = transforms.ToTensor()(orig_img).unsqueeze(0).cuda()

InferenceDetector = config.InferenceDetector.cuda()
preds = get_mmdet_predictions(model, img_tensor, config, InferenceDetector)
if isinstance(preds, tuple):
    preds = preds[0]

print("Type of preds:", type(preds))
if isinstance(preds, list):
    print("Length of preds:", len(preds))
    for i, p in enumerate(preds):
        if p is not None and len(p) > 0:
            print(f"Class {i} pred type: {type(p)}, shape: {getattr(p, 'shape', 'NO SHAPE')}")
            print(f"Class {i} data:\n{p}")
            break

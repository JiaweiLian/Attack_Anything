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
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(config.img_size, config.img_size))

InferenceDetector = config.InferenceDetector.cuda()
preds = get_mmdet_predictions(model, img_tensor, config, InferenceDetector)

# This is the expected fix!
if isinstance(preds, tuple):
    preds = preds[0]
if isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], list):
    preds = preds[0] # remove batch size

for class_idx, class_preds in enumerate(preds):
    for bbox in class_preds:
        if len(bbox) >= 5:
            conf = float(bbox[4])
            if conf > 0.3:
                print(f"Class {class_idx} valid box: conf={conf:.2f} box={bbox[:4]}")

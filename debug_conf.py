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

img_path = "../Datasets/coco/val2017/000000000139.jpg" # just a random coco image
try:
    orig_img = Image.open(img_path).convert('RGB')
except Exception as e:
    import json
    from pycocotools.coco import COCO
    cocoGt = COCO("../Datasets/coco/annotations/instances_val2017.json")
    img_info = cocoGt.loadImgs(list(cocoGt.imgs.keys())[0])[0]
    img_path = "../Datasets/coco/val2017/" + img_info['file_name']
    orig_img = Image.open(img_path).convert('RGB')

img_tensor = transforms.ToTensor()(orig_img).unsqueeze(0).cuda()
img_tensor = torch.nn.functional.interpolate(img_tensor, size=(config.img_size, config.img_size))

from mmdet.apis import inference_detector as InferenceDetector
config.InferenceDetector = InferenceDetector

preds = get_mmdet_predictions(model, img_tensor, config, InferenceDetector)
if isinstance(preds, tuple):
    preds = preds[0]

for class_idx, class_preds in enumerate(preds):
    for bbox in class_preds:
        if len(bbox) >= 5:
            bx1, by1, bx2, by2, conf = float(np.ravel(bbox[0])[0]), float(np.ravel(bbox[1])[0]), float(np.ravel(bbox[2])[0]), float(np.ravel(bbox[3])[0]), float(np.ravel(bbox[4])[0])
            if conf > 0.3:
                print(f"Class: {class_idx}, conf: {conf}, box: {bx1}, {by1}, {bx2}, {by2}")


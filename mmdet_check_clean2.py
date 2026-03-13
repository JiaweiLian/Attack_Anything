import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import evaluate
from tqdm import tqdm

ann_file = '../Datasets/coco/annotations/instances_val2017.json'
img_prefix = '../Datasets/coco/val2017/'
evaluate.COCO = COCO
evaluate.COCOeval = COCOeval
evaluate.tqdm = tqdm

import patch_config
class Args:
    model = "faster_rcnn"
    patch_path = None
    attack_mode = "clean"

# I will just write a custom mini evaluator importing from evaluate.py
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

trainer = evaluate.PatchTrainer("faster_rcnn")
config = trainer.config
model = trainer.model.eval()

cocoGt = COCO(ann_file)
img_ids = cocoGt.getImgIds()[:50]
results = []
InferenceDetector = config.InferenceDetector.cuda()

for idx in img_ids:
    img_info = cocoGt.loadImgs(idx)[0]
    img_path = os.path.join(img_prefix, img_info['file_name'])
    
    image_pil = Image.open(img_path).convert('RGB')
    w, h = image_pil.size
    
    if w == h:
        padded_img = image_pil
        x_offset, y_offset = 0, 0
        pad_scale = config.img_size / w
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(image_pil, (int(padding), 0))
            x_offset, y_offset = int(padding), 0
            pad_scale = config.img_size / h
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(image_pil, (0, int(padding)))
            x_offset, y_offset = 0, int(padding)
            pad_scale = config.img_size / w
            
    padded_img = padded_img.resize((config.img_size, config.img_size))
    img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).cuda()
    
    with torch.no_grad():
        preds = evaluate.get_mmdet_predictions(model, img_tensor, config, InferenceDetector)
        if isinstance(preds, tuple):
            preds = preds[0]
            
        if isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], list):
            preds = preds[0]
            
        for class_idx, class_preds in enumerate(preds):
            if len(class_preds) > 0:
                coco_cat_id = cocoGt.getCatIds()[class_idx]
                for bbox in class_preds:
                    if len(bbox) >= 5:
                        x1, y1, x2, y2, conf = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
                        x1 = (x1 / pad_scale) - x_offset
                        x2 = (x2 / pad_scale) - x_offset
                        y1 = (y1 / pad_scale) - y_offset
                        y2 = (y2 / pad_scale) - y_offset
                        width, height = x2 - x1, y2 - y1
                        results.append({
                            "image_id": idx,
                            "category_id": coco_cat_id,
                            "bbox": [float(x1), float(y1), float(width), float(height)],
                            "score": float(conf)
                        })

temp_json = 'temp_mini.json'
with open(temp_json, 'w') as f:
    json.dump(results, f)
cocoDt = cocoGt.loadRes(temp_json)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = img_ids
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

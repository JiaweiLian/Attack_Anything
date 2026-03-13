from evaluate import get_mmdet_predictions, PatchTrainer
import torch
import numpy as np

trainer = PatchTrainer('mask_rcnn')
model = trainer.model.eval()
config = trainer.config

img_tensor = torch.zeros((1, 3, 1024, 1024)).cuda()
preds = get_mmdet_predictions(model, img_tensor, config, config.InferenceDetector)
print("Type of preds:", type(preds))
if isinstance(preds, list):
    print("preds len:", len(preds))
    print("Type of preds[0]:", type(preds[0]))
    if isinstance(preds[0], tuple):
        print("is tuple, len:", len(preds[0]))
        print("Type of preds[0][0]:", type(preds[0][0]))
        if isinstance(preds[0][0], list):
            print("is list, len:", len(preds[0][0]))
            print("Type of preds[0][0][0]:", type(preds[0][0][0]))
            print("Shape:", getattr(preds[0][0][0], 'shape', 'No shape'))

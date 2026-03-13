import argparse
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import patch_config
import torch.nn.functional as F
import numpy as np

config = patch_config.patch_configs['faster_rcnn']()
from train import PatchTrainer
trainer = PatchTrainer('faster_rcnn')
model = trainer.model.eval()

img_path = "../Datasets/coco/val2017/000000000139.jpg"
orig_img = Image.open(img_path).convert('RGB')
w, h = orig_img.size

if w == h:
    padded_img = orig_img
else:
    dim_to_pad = 1 if w < h else 2
    if dim_to_pad == 1:
        padding = (h - w) / 2
        padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
        padded_img.paste(orig_img, (int(padding), 0))
    else:
        padding = (w - h) / 2
        padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
        padded_img.paste(orig_img, (0, int(padding)))
        
padded_img = padded_img.resize((config.img_size, config.img_size))
img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).cuda()

def get_preds(model, img_t, rescale):
    InferenceDetector = config.InferenceDetector.cuda()
    img_cpu = img_t[0].detach().cpu().numpy()
    img_cpu = img_cpu.transpose(1, 2, 0) * 255.0
    data = InferenceDetector(model, img_cpu)
    
    # Let's inspect data
    print("img_metas:", data['img_metas'][0])
    
    data['img'][0] = img_t
    return model(return_loss=False, rescale=rescale, **data)

print("With rescale=False")
preds_false = get_preds(model, img_tensor, False)
print(preds_false[0][0])

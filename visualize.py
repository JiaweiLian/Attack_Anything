import argparse
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from pycocotools.coco import COCO
import patch_config
import torch.nn.functional as F
import numpy as np
import os
import warnings
from train import adv_patch_update, PatchTrainer
from evaluate import get_yolo_predictions, get_mmdet_predictions
from tqdm import tqdm

def visualize(model_name, patch_path=None, attack_mode_override=None, num_images=20):
    config = patch_config.patch_configs[model_name]()
    if attack_mode_override:
        config.attack_mode = attack_mode_override

    trainer = PatchTrainer(model_name)
    model = trainer.model.eval()
    
    warnings.filterwarnings("ignore")

    # 2. Load Dataset
    val_json = "../Datasets/coco/annotations/instances_val2017.json"
    val_img_dir = "../Datasets/coco/val2017/"
    cocoGt = COCO(val_json)
    
    # 3. Load Patch
    adv_patch = None
    if patch_path and os.path.exists(patch_path):
        patch_img = Image.open(patch_path).convert('RGB')
        adv_patch = transforms.ToTensor()(patch_img).cuda().unsqueeze(0)
    
    out_dir = "clean_vis_results" if adv_patch is None else "vis_results"
    os.makedirs(out_dir, exist_ok=True)
    img_ids = list(cocoGt.imgs.keys())
    
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    cat_map = {c['id']: c['name'] for c in cats}

    # Font handling
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    print(f"Generating {num_images} visualizations...")
    for i in tqdm(range(num_images)):
        img_id = img_ids[i]
        img_info = cocoGt.loadImgs(img_id)[0]
        img_path = os.path.join(val_img_dir, img_info['file_name'])
        
        orig_img = Image.open(img_path).convert('RGB')
        w, h = orig_img.size
        
        # New Padding Logic instead of stretching
        if w == h:
            padded_img = orig_img
            x_offset, y_offset = 0, 0
            pad_scale = config.img_size / w
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(orig_img, (int(padding), 0))
                x_offset, y_offset = int(padding), 0
                pad_scale = config.img_size / h
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(orig_img, (0, int(padding)))
                x_offset, y_offset = 0, int(padding)
                pad_scale = config.img_size / w
                
        padded_img = padded_img.resize((config.img_size, config.img_size))
        img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).cuda()
        
        if adv_patch is not None:
            ann_ids = cocoGt.getAnnIds(imgIds=img_id)
            anns = cocoGt.loadAnns(ann_ids)
            
            mask_batch = torch.zeros((1, 1, config.img_size, config.img_size)).cuda()
            for ann in anns:
                bbox = ann['bbox'] # [x, y, width, height]
                x1 = int((bbox[0] + x_offset) * pad_scale)
                y1 = int((bbox[1] + y_offset) * pad_scale)
                x2 = int((bbox[0] + bbox[2] + x_offset) * pad_scale)
                y2 = int((bbox[1] + bbox[3] + y_offset) * pad_scale)
                # Keep within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(config.img_size-1, x2), min(config.img_size-1, y2)
                mask_batch[0, 0, y1:y2, x1:x2] = 1.0
                
            adv_patch_resized = F.interpolate(adv_patch, size=(config.img_size, config.img_size))
            
            if getattr(config, 'attack_mode', 'tba') == 'bba':
                H_img, W_img = config.img_size, config.img_size
                strip_mask = torch.zeros_like(mask_batch)
                strip_mask[:, :, H_img // 4 : 3 * H_img // 4, :] = 1.0
                bba_mask = strip_mask * (1.0 - mask_batch)
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, 1.0 - bba_mask, bba_mask)
            else:
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, mask_batch, 1.0 - mask_batch)
        else:
            input_tensor = img_tensor
            
        # Get image back to CPU for drawing
        vis_img = input_tensor[0].cpu().permute(1, 2, 0).numpy() * 255.0
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        pil_vis_img = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(pil_vis_img)
            
        with torch.no_grad():
            if trainer.yolo:
                preds = get_yolo_predictions(model, input_tensor, config)[0]
                if preds is not None and len(preds):
                    for det in preds:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        if conf > 0.3:
                            coco_cat_id = cocoGt.getCatIds()[int(cls)] 
                            cat_name = cat_map[coco_cat_id] if coco_cat_id in cat_map else str(int(cls))
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                            text_str = f"{cat_name} {conf:.2f}"
                            
                            # Add background for text to make it readable
                            try:
                                text_bbox = font.getbbox(text_str) if hasattr(font, "getbbox") else font.getsize(text_str)
                                text_w = text_bbox[2] - text_bbox[0] if len(text_bbox) == 4 else text_bbox[0]
                                text_h = text_bbox[3] - text_bbox[1] if len(text_bbox) == 4 else text_bbox[1]
                            except:
                                text_w, text_h = 60, 15
                            draw.rectangle([x1, max(0, y1-text_h), x1+text_w, y1], fill="red")
                            draw.text((x1, max(0, y1-text_h)), text_str, fill="white", font=font)
            else:
                InferenceDetector = config.InferenceDetector.cuda() if hasattr(config, 'InferenceDetector') else None
                preds = get_mmdet_predictions(model, input_tensor, config, InferenceDetector)
                if isinstance(preds, list) and len(preds) == 1:
                    preds = preds[0]
                if isinstance(preds, tuple):
                    preds = preds[0]
                for class_idx, class_preds in enumerate(preds):
                    coco_cat_id = cocoGt.getCatIds()[class_idx]
                    cat_name = cat_map[coco_cat_id] if coco_cat_id in cat_map else str(class_idx)
                    for bbox in class_preds:
                        if len(bbox) >= 5:
                            bx1, by1, bx2, by2, conf = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
                            if conf > 0.3:
                                # Coordinates are already in padded square size natively
                                draw.rectangle([bx1, by1, bx2, by2], outline="red", width=4)
                                text_str = f"{cat_name} {conf:.2f}"
                                
                                try:
                                    text_bbox = font.getbbox(text_str) if hasattr(font, "getbbox") else font.getsize(text_str)
                                    text_w = text_bbox[2] - text_bbox[0] if len(text_bbox) == 4 else text_bbox[0]
                                    text_h = text_bbox[3] - text_bbox[1] if len(text_bbox) == 4 else text_bbox[1]
                                except:
                                    text_w, text_h = 60, 15
                                draw.rectangle([bx1, max(0, by1-text_h), bx1+text_w, by1], fill="red")
                                draw.text((bx1, max(0, by1-text_h)), text_str, fill="white", font=font)
        
        pil_vis_img.save(f"{out_dir}/visualization_{i:02d}.jpg")

if __name__ == '__main__':
    # visualize('ssd', 'patches/patch_NN_response/bba_faster_rcnn.png', 'bba', 20)
    visualize('vfnet', None, 'bba', 20)

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
from evaluate import get_yolo_predictions, get_mmdet_predictions, apply_viewpoint_change
from tqdm import tqdm
import argparse

def visualize(model_name, patch_path=None, attack_mode_override=None, num_images=20, mask_noise_type='none', mask_noise_val=0, viewpoint=0):
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
        patch_name = os.path.splitext(os.path.basename(patch_path))[0]
    else:
        patch_name = "clean"
        
    if mask_noise_type != 'none' and mask_noise_val > 0:
        patch_name = f"{patch_name}_{mask_noise_type}_{mask_noise_val}"
    
    if viewpoint != 0:
        patch_name = f"{patch_name}_vp{viewpoint}"
        
    out_dir = os.path.join("visual_results", f"{model_name}_{patch_name}")
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
            attack_mode = getattr(config, 'attack_mode', 'tba')
            for ann in anns:
                if attack_mode in ['tba', 'ccba'] and getattr(config, 'use_mask', True):
                    try:
                        mask = cocoGt.annToMask(ann)
                        mask_pil = Image.fromarray(mask * 255)
                        
                        if h > w:
                            padded_mask = Image.new('L', (h, h), 0)
                        else:
                            padded_mask = Image.new('L', (w, w), 0)
                            
                        padded_mask.paste(mask_pil, (x_offset, y_offset))
                        padded_mask = padded_mask.resize((config.img_size, config.img_size), resample=Image.NEAREST)
                        
                        # Apply experimental noise to mask for visualization
                        if mask_noise_type != 'none' and mask_noise_val > 0:
                            if mask_noise_type in ['dilate', 'erode']:
                                import cv2
                                mask_np = np.array(padded_mask)
                                kernel_size = 1 + 2 * mask_noise_val
                                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                                if mask_noise_type == 'dilate':
                                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                                else:
                                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
                                padded_mask = Image.fromarray(mask_np)
                            elif mask_noise_type == 'shift':
                                padded_mask = padded_mask.transform(padded_mask.size, Image.AFFINE, (1, 0, mask_noise_val, 0, 1, mask_noise_val))
                            elif mask_noise_type == 'flip':
                                mask_np = np.array(padded_mask)
                                # randomly flip mask_noise_val% of pixels inside the bounding box of the mask
                                flip_mask = np.random.rand(*mask_np.shape) < (mask_noise_val / 100.0)
                                mask_np = np.logical_xor(mask_np > 0, flip_mask).astype(np.uint8) * 255
                                padded_mask = Image.fromarray(mask_np)

                        mask_tensor = transforms.ToTensor()(padded_mask).cuda()
                        mask_batch[0, 0] = torch.max(mask_batch[0, 0], mask_tensor[0])
                    except Exception as e:
                        bbox = ann['bbox'] # [x, y, width, height]
                        x1 = int((bbox[0] + x_offset) * pad_scale)
                        y1 = int((bbox[1] + y_offset) * pad_scale)
                        x2 = int((bbox[0] + bbox[2] + x_offset) * pad_scale)
                        y2 = int((bbox[1] + bbox[3] + y_offset) * pad_scale)
                        # Keep within bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(config.img_size-1, x2), min(config.img_size-1, y2)
                        mask_batch[0, 0, y1:y2, x1:x2] = 1.0
                else:
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
            
            adv_patch_resized, alpha_vp = apply_viewpoint_change(adv_patch_resized, viewpoint)
            
            attack_mode = getattr(config, 'attack_mode', 'tba')
            if attack_mode == 'bba':
                H_img, W_img = config.img_size, config.img_size
                strip_mask = torch.zeros_like(mask_batch)
                strip_mask[:, :, H_img // 4 : 3 * H_img // 4, :] = 1.0
                bba_mask = strip_mask * (1.0 - mask_batch) * alpha_vp
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, 1.0 - bba_mask, bba_mask)
            elif attack_mode == 'ccba':
                H_img, W_img = config.img_size, config.img_size
                y_grid, x_grid = torch.meshgrid(torch.arange(H_img), torch.arange(W_img))
                dist = torch.sqrt((x_grid - W_img//2)**2 + (y_grid - H_img//2)**2).to(img_tensor.device)
                ring_width = 60 # Coarser rings conforming to characteristic size in Reference 1
                
                # Calculate unique index for each ring
                ring_idx = (dist / ring_width).long()
                max_rings = int((W_img / 2) / ring_width) + 2
                
                b_size = adv_patch_resized.size(0)
                radial_patch = torch.zeros_like(adv_patch_resized)
                for r in range(max_rings):
                    mask_r = (ring_idx == r).unsqueeze(0).unsqueeze(0).float()
                    count = mask_r.sum()
                    if count > 0:
                        sum_color = (adv_patch_resized * mask_r).sum(dim=(2, 3), keepdim=True)
                        mean_color = sum_color / count
                        radial_patch += mean_color * mask_r

                # Limit to maximum diameter = img_size (e.g., 1024)
                valid_ring_mask = (dist <= W_img/2).float()
                ccba_mask = valid_ring_mask.unsqueeze(0).unsqueeze(0).expand_as(mask_batch) * (1.0 - mask_batch) * alpha_vp
                
                input_tensor = adv_patch_update(radial_patch, img_tensor, 1.0 - ccba_mask, ccba_mask)
            else: # tba
                tba_mask = (1.0 - mask_batch) * alpha_vp
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, 1.0 - tba_mask, tba_mask)
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
    parser = argparse.ArgumentParser(description='Visualize Adversarial Patch on Image')
    parser.add_argument('--model', type=str, default="yolov5x", help='Target detector mode (e.g. ssd, yolov5m).')
    parser.add_argument('--patch_path', type=str, default="None", help='Path to the trained adversarial patch image. If "None", visualizes clean image.')
    parser.add_argument('--attack_mode', type=str, default="bba", choices=['tba', 'bba', 'ccba', 'clean'], help='Attack mode to visualize.')
    parser.add_argument('--num_images', type=int, default=20, help='Number of images to visualize.')
    parser.add_argument('--mask_noise_type', type=str, default='none', choices=['none', 'dilate', 'erode', 'flip', 'shift'], help='Type of noise applied to the mask.')
    parser.add_argument('--mask_noise_val', type=int, default=0, help='Value for the mask noise.')
    parser.add_argument('--viewpoint', type=int, default=0, help='Viewpoint angle in degrees (e.g., 45 for side view).')
    
    args = parser.parse_args()
    
    # Handle the string "None" passed from command line
    patch_path = None if args.patch_path.lower() == "none" else args.patch_path
    
    visualize(args.model, patch_path, args.attack_mode, args.num_images, args.mask_noise_type, args.mask_noise_val, args.viewpoint)

# python visualize.py --model yolov5x --patch_path patches/patch_NN_response/ccba_yolov5x.png --attack_mode ccba --num_images 100

# CUDA_VISIBLE_DEVICES=1 python visualize.py --model yolov5x --attack_mode tba --patch_path patches/patch_NN_response/tba_yolov5x.png --mask_noise_type dilate --mask_noise_val 32 --num_images 100

# CUDA_VISIBLE_DEVICES=1 python visualize.py --model yolov5x --attack_mode tba --patch_path patches/patch_NN_response/tba_yolov5x_tood_tv.png --num_images 100 --viewpoint 0

import argparse
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import patch_config
import torch.nn.functional as F
import numpy as np
import json
import os
import warnings
from train import adv_patch_update, PatchTrainer

warnings.filterwarnings('ignore')

def apply_viewpoint_change(patch_tensor, angle):
    """Applies a perspective transform to the patch to simulate viewing from a side angle.
    Returns the transformed patch and a binary alpha mask indicating valid pixels."""
    if angle == 0:
        return patch_tensor, torch.ones((1, 1, patch_tensor.shape[-2], patch_tensor.shape[-1]), device=patch_tensor.device)
        
    import torchvision.transforms.functional as TF
    import math
    
    _, _, H, W = patch_tensor.shape
    alpha_mask = torch.ones((1, 1, H, W), device=patch_tensor.device)
    
    startpoints = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
    
    angle_rad = math.radians(abs(angle))
    shrink_h = int((H / 2) * math.sin(angle_rad))
    shrink_w = int((W / 2) * (1 - math.cos(angle_rad)))
    
    if angle > 0: 
        endpoints = [
            [0 + shrink_w, 0 + shrink_h], 
            [W - 1, 0], 
            [W - 1, H - 1], 
            [0 + shrink_w, H - 1 - shrink_h]
        ]
    else:
        endpoints = [
            [0, 0], 
            [W - 1 - shrink_w, 0 + shrink_h], 
            [W - 1 - shrink_w, H - 1 - shrink_h], 
            [0, H - 1]
        ]
        
    patch_t = TF.perspective(patch_tensor, startpoints, endpoints)
    alpha_t = TF.perspective(alpha_mask, startpoints, endpoints)
    
    alpha_t = (alpha_t > 0.5).float()
    return patch_t, alpha_t

def load_patch(patch_path, config):
    """Load and transform the trained patch."""
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    patch_img = Image.open(patch_path).convert('RGB')
    tf = transforms.Resize((config.patch_size, config.patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    return adv_patch_cpu.unsqueeze(0).cuda()  # [1, 3, H, W]

def get_yolo_predictions(model, img_tensor, config):
    """Get bounding box predictions from YOLO model."""
    # Assuming standard predict output format from your prob_extractor or inference tools
    output = model(img_tensor)
    from utils_yolov5.general import non_max_suppression # using yolov5 utils you have
    preds = non_max_suppression(output, config.conf_thres, config.iou_thres, classes=config.classes, agnostic=config.agnostic_nms, max_det=config.max_det)
    return preds

def get_mmdet_predictions(model, img_tensor, config, InferenceDetector):
    # mmdet expects normalized tensor, img_tensor is currently 0-1
    # Check if model has img_norm_cfg, else fallback to standard ImageNet
    mean_val = [123.675, 116.28, 103.53]
    std_val = [58.395, 57.12, 57.375]
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'img_norm_cfg'):
        norm_cfg = model.cfg.img_norm_cfg
        mean_val = norm_cfg.get('mean', mean_val)
        std_val = norm_cfg.get('std', std_val)
        
    mean = torch.tensor(mean_val, device=img_tensor.device).view(1, 3, 1, 1) / 255.0
    std = torch.tensor(std_val, device=img_tensor.device).view(1, 3, 1, 1) / 255.0
    
    norm_tensor = (img_tensor - mean) / std
    
    from mmcv.parallel import DataContainer
    data = dict(
        img=[norm_tensor],
        img_metas=[[dict(
            ori_shape=(config.img_size, config.img_size, 3),
            img_shape=(config.img_size, config.img_size, 3),
            pad_shape=(config.img_size, config.img_size, 3),
            scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            flip=False,
        )]]
    )
    
    output = model(return_loss=False, rescale=False, **data)
    return output

def evaluate_map(mode, patch_path=None, attack_mode_override=None, split='val', mask_noise_type='none', mask_noise_val=0, viewpoint=0):
    """
    Evaluate the mAP of a given model and patch on the dataset.
    """
    trainer = PatchTrainer(mode)
    config = trainer.config
    
    if attack_mode_override is not None:
        config.attack_mode = attack_mode_override
        
    model = trainer.model.eval()
    
    # We use validation set for actual evaluation
    ann_file = '../Datasets/coco/annotations/instances_val2017.json'  # Need to ensure path
    img_prefix = '../Datasets/coco/val2017/'
    
    print(f"Loading COCO Annotations from {ann_file}")
    cocoGt = COCO(ann_file)
    img_ids = cocoGt.getImgIds()
    
    # Load patch if provided
    has_patch = patch_path is not None
    adv_patch = None
    if has_patch:
        print(f"Loading Adversarial Patch from {patch_path} (Mode: {getattr(config, 'attack_mode', 'tba')})")
        adv_patch = load_patch(patch_path, config)
    else:
        print("Evaluating Clean Original Performance...")
        
    results = []
    
    print(f"Starting Evaluation on {len(img_ids)} images...")
    for idx in tqdm(img_ids):
        img_info = cocoGt.loadImgs(idx)[0]
        img_path = os.path.join(img_prefix, img_info['file_name'])
        
        # 1. Load Image
        image_pil = Image.open(img_path).convert('RGB')
        w, h = image_pil.size
        
        # 1.1 Pad image to square (127, 127, 127) as in train.py DataLoader
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

        # 2. Resize and convert to tensor
        padded_img = padded_img.resize((config.img_size, config.img_size))
        img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).cuda()
        
        # 3. Apply Patch
        if has_patch and getattr(config, 'attack_mode', 'tba') != 'clean':
            # We need to construct a mask based on GT for applying the patch 
            # (assuming attack needs to avoid GT boxes just like training)
            ann_ids = cocoGt.getAnnIds(imgIds=idx)
            anns = cocoGt.loadAnns(ann_ids)
            
            mask_batch = torch.zeros((1, 1, config.img_size, config.img_size)).cuda()
            for ann in anns:
                bbox = ann['bbox'] # [x, y, width, height] in original image scale
                # Scale box to img_size
                x1 = int((bbox[0] + x_offset) * pad_scale)
                y1 = int((bbox[1] + y_offset) * pad_scale)
                x2 = int((bbox[0] + bbox[2] + x_offset) * pad_scale)
                y2 = int((bbox[1] + bbox[3] + y_offset) * pad_scale)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(config.img_size-1, x2), min(config.img_size-1, y2)
                if x2 > x1 and y2 > y1:
                    mask_batch[0, 0, y1:y2, x1:x2] = 1.0 # 1 inside box
            
            # Apply experimental noise to mask for evaluation sensitivity analysis
            if mask_noise_type != 'none' and mask_noise_val > 0:
                mask_np = mask_batch[0, 0].cpu().numpy().astype(np.uint8) * 255
                if mask_noise_type in ['dilate', 'erode']:
                    import cv2
                    kernel_size = 1 + 2 * mask_noise_val
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    if mask_noise_type == 'dilate':
                        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                    else:
                        mask_np = cv2.erode(mask_np, kernel, iterations=1)
                elif mask_noise_type == 'shift':
                    import cv2
                    M = np.float32([[1, 0, mask_noise_val], [0, 1, mask_noise_val]])
                    mask_np = cv2.warpAffine(mask_np, M, (config.img_size, config.img_size))
                elif mask_noise_type == 'flip':
                    flip_mask = np.random.rand(*mask_np.shape) < (mask_noise_val / 100.0)
                    mask_np = np.logical_xor(mask_np > 0, flip_mask).astype(np.uint8) * 255
                mask_batch[0, 0] = torch.from_numpy(mask_np).cuda().float() / 255.0
            
            adv_patch_resized = F.interpolate(adv_patch, size=(config.img_size, config.img_size))
            
            # Apply viewpoint transformation
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

        # 4. Inference
        with torch.no_grad():
            if trainer.yolo:
                preds = get_yolo_predictions(model, input_tensor, config)[0]
                # Format: [x1, y1, x2, y2, conf, cls]
                if preds is not None and len(preds):
                    for det in preds:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        # rescale back
                        x1 = (x1 / pad_scale) - x_offset
                        x2 = (x2 / pad_scale) - x_offset
                        y1 = (y1 / pad_scale) - y_offset
                        y2 = (y2 / pad_scale) - y_offset
                        width, height = x2 - x1, y2 - y1
                        
                        coco_cat_id = cocoGt.getCatIds()[int(cls)] 
                        results.append({
                            "image_id": idx,
                            "category_id": coco_cat_id,
                            "bbox": [float(x1), float(y1), float(width), float(height)],
                            "score": float(conf)
                        })
            else:
                InferenceDetector = config.InferenceDetector.cuda() if hasattr(config, 'InferenceDetector') else None
                preds = get_mmdet_predictions(model, input_tensor, config, InferenceDetector)
                
                # Remove batch dimension if it exists
                if isinstance(preds, list) and len(preds) == 1:
                    preds = preds[0]
                    
                # MMDetection often returns a tuple of (bbox_results, segm_results) for instance segmentation networks
                if isinstance(preds, tuple):
                    preds = preds[0]
                    
                for class_idx, class_preds in enumerate(preds):
                    if len(class_preds) > 0:
                        coco_cat_id = cocoGt.getCatIds()[class_idx]
                        for bbox in class_preds:
                            # Safely ignore empty sub-arrays
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

    if not results:
        print("No objects detected. AP is 0.")
        return

    # 5. Evaluate
    temp_json = f'temp_results_{mode}.json'
    with open(temp_json, 'w') as f:
        json.dump(results, f)
        
    cocoDt = cocoGt.loadRes(temp_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    # We care about mAP@0.5 heavily
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print("\n=============================================")
    print("                Summary Map                  ")
    print(f"AP@0.5:0.95 = {cocoEval.stats[0]:.4f}")
    print(f"AP@0.5      = {cocoEval.stats[1]:.4f}  <-- Target Metric")
    print("=============================================\n")
    
    if os.path.exists(temp_json):
        os.remove(temp_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Adversarial Patch mAP')
    parser.add_argument('--model', type=str, default="yolov5x", help='Target detector mode (e.g. ssd, yolov5m).')
    parser.add_argument('--patch_path', type=str, default="patches/patch_NN_response/tba_yolov5x.png", help='Path to the trained adversarial patch image. If None, evaluates clean image.')
    parser.add_argument('--attack_mode', type=str, default="tba", choices=['tba', 'bba', 'ccba', 'clean'], help='Override the attack mode set in patch_config (tba, bba, or ccba).')
    parser.add_argument('--mask_noise_type', type=str, default='none', choices=['none', 'dilate', 'erode', 'flip', 'shift'], help='Type of noise applied to the mask at evaluation time.')
    parser.add_argument('--mask_noise_val', type=int, default=0, help='Value for the mask noise (e.g. kernel size for dilation/erosion, or pixel shift, or flip percentage).')
    parser.add_argument('--viewpoint', type=int, default=0, help='Viewpoint angle in degrees (e.g., 45 for side view).')
    args = parser.parse_args()
    
    if args.model is None:
        args.model = patch_config.BaseConfig().target_detector
        
    config = patch_config.patch_configs[args.model]()
    if args.attack_mode is not None:
        config.attack_mode = args.attack_mode
        
    evaluate_map(args.model, args.patch_path, attack_mode_override=args.attack_mode, 
                 mask_noise_type=args.mask_noise_type, mask_noise_val=args.mask_noise_val,
                 viewpoint=args.viewpoint)

# CUDA_VISIBLE_DEVICES=0 python evaluate.py --model yolov5x --attack_mode tba --patch_path patches/patch_NN_response/tba_yolov5x.png --mask_noise_type dilate --mask_noise_val 32

# CUDA_VISIBLE_DEVICES=2 python evaluate.py --model yolov5x --attack_mode tba --patch_path patches/patch_NN_response/tba_yolov5x_tood_tv_ensemble.png --viewpoint 4

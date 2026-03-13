import sys

with open('evaluate.py', 'r') as f:
    eval_code = f.read()

old_loop = """    print(f"Starting Evaluation on {len(img_ids)} images...")
    for idx in tqdm(img_ids):
        img_info = cocoGt.loadImgs(idx)[0]
        img_path = os.path.join(img_prefix, img_info['file_name'])
        
        # 1. Load Image
        image_pil = Image.open(img_path).convert('RGB')
        w, h = image_pil.size
        
        # 2. Resize and Pad image to config.img_size as in train.py DataLoader
        tf = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)), 
            transforms.ToTensor()
        ])
        img_tensor = tf(image_pil).unsqueeze(0).cuda()
        
        # 3. Apply Patch
        if has_patch:
            # We need to construct a mask based on GT for applying the patch 
            # (assuming attack needs to avoid GT boxes just like training)
            ann_ids = cocoGt.getAnnIds(imgIds=idx)
            anns = cocoGt.loadAnns(ann_ids)
            
            mask_batch = torch.zeros((1, 1, config.img_size, config.img_size)).cuda()
            for ann in anns:
                bbox = ann['bbox'] # [x, y, width, height] in original image scale
                # Scale box to img_size
                x1 = int(bbox[0] * config.img_size / w)
                y1 = int(bbox[1] * config.img_size / h)
                x2 = int((bbox[0] + bbox[2]) * config.img_size / w)
                y2 = int((bbox[1] + bbox[3]) * config.img_size / h)
                mask_batch[0, 0, y1:y2, x1:x2] = 1.0 # 1 inside box
            
            adv_patch_resized = F.interpolate(adv_patch, size=(config.img_size, config.img_size))
            
            if getattr(config, 'attack_mode', 'tba') == 'bba':
                H_img, W_img = config.img_size, config.img_size
                strip_mask = torch.zeros_like(mask_batch)
                strip_mask[:, :, H_img // 4 : 3 * H_img // 4, :] = 1.0
                bba_mask = strip_mask * (1.0 - mask_batch)
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, 1.0 - bba_mask, bba_mask)
            else: # tba
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, mask_batch, 1.0 - mask_batch)
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
                        x1, x2 = x1 * w / config.img_size, x2 * w / config.img_size
                        y1, y2 = y1 * h / config.img_size, y2 * h / config.img_size
                        width, height = x2 - x1, y2 - y1
                        # Map internal class ID back to COCO category id
                        # Assuming direct mapping if you are using default 80 classes, else needs translation
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
                
                # MMDetection often returns a tuple of (bbox_results, segm_results) for instance segmentation networks
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                # Remove batch dimension if it exists
                if isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], list):
                    preds = preds[0]
                    
                for class_idx, class_preds in enumerate(preds):
                    if len(class_preds) > 0:
                        coco_cat_id = cocoGt.getCatIds()[class_idx]
                        for bbox in class_preds:
                            # Safely ignore empty sub-arrays
                            if len(bbox) >= 5:
                                x1, y1, x2, y2, conf = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
                                    
                                x1, x2 = float(x1) * w / config.img_size, float(x2) * w / config.img_size
                                y1, y2 = float(y1) * h / config.img_size, float(y2) * h / config.img_size
                                width, height = x2 - x1, y2 - y1
                                results.append({
                                    "image_id": idx,
                                    "category_id": coco_cat_id,
                                    "bbox": [float(x1), float(y1), float(width), float(height)],
                                    "score": float(conf)
                                })"""

new_loop = """    print(f"Starting Evaluation on {len(img_ids)} images...")
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
            
            adv_patch_resized = F.interpolate(adv_patch, size=(config.img_size, config.img_size))
            
            if getattr(config, 'attack_mode', 'tba') == 'bba':
                H_img, W_img = config.img_size, config.img_size
                strip_mask = torch.zeros_like(mask_batch)
                strip_mask[:, :, H_img // 4 : 3 * H_img // 4, :] = 1.0
                bba_mask = strip_mask * (1.0 - mask_batch)
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, 1.0 - bba_mask, bba_mask)
            else: # tba
                input_tensor = adv_patch_update(adv_patch_resized, img_tensor, mask_batch, 1.0 - mask_batch)
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
                
                # MMDetection often returns a tuple of (bbox_results, segm_results) for instance segmentation networks
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                # Remove batch dimension if it exists
                if isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], list):
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
                                })"""

if old_loop in eval_code:
    eval_code = eval_code.replace(old_loop, new_loop)
    with open('evaluate.py', 'w') as f:
        f.write(eval_code)
    print("Evaluate logic successfully replaced!")
else:
    print("Could not find the old loop string exactly.")


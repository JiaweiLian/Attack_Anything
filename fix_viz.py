import re

with open('visualize.py', 'r') as f:
    text = f.read()

# Make sure visualize.py normalizes before predicting!
fix_code = """
            if trainer.yolo:
                preds = get_yolo_predictions(model, input_tensor, config)[0]
                if preds is not None and len(preds):
                    for det in preds:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
                        
                        coco_cat_id = cocoGt.getCatIds()[int(cls)]
                        if coco_cat_id in cocoGt.cats:
                            cat_name = cocoGt.cats[coco_cat_id]['name']
                        else:
                            cat_name = str(int(cls))
                            
                        label_str = f"{cat_name}: {conf:.2f}"
                        text_bbox = draw.textbbox((x1, y1 - 25), label_str, font=font)
                        draw.rectangle(text_bbox, fill="red")
                        draw.text((x1, y1 - 25), label_str, fill="white", font=font)
            else:
                InferenceDetector = config.InferenceDetector.cuda() if hasattr(config, 'InferenceDetector') else None
                
                # MMDetection normalizer
                mean = torch.tensor([123.675, 116.28, 103.53], device=input_tensor.device).view(1, 3, 1, 1) / 255.0
                std = torch.tensor([58.395, 57.12, 57.375], device=input_tensor.device).view(1, 3, 1, 1) / 255.0
                norm_tensor = (input_tensor - mean) / std
                
                data = dict(
                    img=[norm_tensor],
                    img_metas=[[dict(
                        ori_shape=(config.img_size, config.img_size, 3),
                        img_shape=(config.img_size, config.img_size, 3),
                        pad_shape=(config.img_size, config.img_size, 3),
                        scale_factor=1.0,
                        flip=False,
                    )]]
                )
                preds = model(return_loss=False, rescale=False, **data)
"""

find_str = """
            if trainer.yolo:
                preds = get_yolo_predictions(model, input_tensor, config)[0]
                if preds is not None and len(preds):
                    for det in preds:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
                        
                        coco_cat_id = cocoGt.getCatIds()[int(cls)]
                        if coco_cat_id in cocoGt.cats:
                            cat_name = cocoGt.cats[coco_cat_id]['name']
                        else:
                            cat_name = str(int(cls))
                            
                        label_str = f"{cat_name}: {conf:.2f}"
                        text_bbox = draw.textbbox((x1, y1 - 25), label_str, font=font)
                        draw.rectangle(text_bbox, fill="red")
                        draw.text((x1, y1 - 25), label_str, fill="white", font=font)
            else:
                InferenceDetector = config.InferenceDetector.cuda() if hasattr(config, 'InferenceDetector') else None
                preds = get_mmdet_predictions(model, input_tensor, config, InferenceDetector)
"""

if find_str in text:
    text = text.replace(find_str, fix_code)
    with open('visualize.py', 'w') as f:
        f.write(text)
    print("viz fixed")
else:
    print("could not find block in viz")

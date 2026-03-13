import re

with open('evaluate.py', 'r') as f:
    text = f.read()

# InferenceDetector pipeline also resizes the image to 1333x800 for faster rcnn!
# If we do rescale=False, the boxes are returned in 1333x800 scale, NOT 416x416!
# That's why the mAP was so low: they were scaled wrongly entirely!
# If we want the pipeline normalization BUT our 416x416 boxes, we should write our own normalization inside get_mmdet_predictions
# and NOT use InferenceDetector pipeline.

fix_code = """
def get_mmdet_predictions(model, img_tensor, config, InferenceDetector):
    # mmdet expects normalized tensor, img_tensor is currently 0-1
    # For coco models it's mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    # But those are for 0-255 images. Since img_tensor is 0-1, we divide those by 255.
    
    mean = torch.tensor([123.675, 116.28, 103.53], device=img_tensor.device).view(1, 3, 1, 1) / 255.0
    std = torch.tensor([58.395, 57.12, 57.375], device=img_tensor.device).view(1, 3, 1, 1) / 255.0
    
    norm_tensor = (img_tensor - mean) / std
    
    from mmcv.parallel import DataContainer
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
    
    output = model(return_loss=False, rescale=False, **data)
    return output
"""

find_str = """def get_mmdet_predictions(model, img_tensor, config, InferenceDetector):
    \"\"\"Get bounding box predictions from MMDetection model.\"\"\"
    img_cpu = img_tensor[0].detach().cpu().numpy()
    if img_cpu.shape[0] == 3:
        img_cpu = img_cpu.transpose(1, 2, 0)
    img_cpu = img_cpu * 255.0
    
    data = InferenceDetector(model, img_cpu)
    # # data['img'][0] = img_tensor # REMOVED: let the pipeline keep its normalized tensor! # REMOVED: let the pipeline keep its normalized tensor!
    output = model(return_loss=False, rescale=False, **data)
    # output is a list of lists of shape (N, 5) representing [N_class, N_bboxes, [x1, y1, x2, y2, score]]
    return output"""

if find_str in text:
    text = text.replace(find_str, fix_code.strip())
    with open('evaluate.py', 'w') as f:
        f.write(text)
    print("Fixed logic for mmdet manual inference!")
else:
    print("Could not find the function to replace")

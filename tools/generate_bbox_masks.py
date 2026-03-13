import os
import cv2
import numpy as np

def generate_masks(img_dir, label_dir, output_dir):
    """
    根据给定的图像和 txt 标签（YOLO格式），生成一一对应的二值化目标框掩码。
    :param img_dir: 原始图像文件夹路径
    :param label_dir: txt标签文件夹路径（通常为类编号、归一化的 x_center, y_center, w, h）
    :param output_dir: 掩码图像输出路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"新建输出文件夹: {output_dir}")

    # 支持的图像格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 获取图片列表
    images = []
    if not os.path.exists(img_dir):
        print(f"错误: 找不到图片文件夹 {img_dir}")
        return
        
    for f in os.listdir(img_dir):
        if f.lower().endswith(img_extensions):
            images.append(f)
            
    print(f"正在读取... 在 {img_dir} 中找到了 {len(images)} 张图片。")

    success_count = 0
    empty_label_count = 0
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # 1. 读取原图以获取其狂高分辨率
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}，跳过该图。")
            continue
            
        height, width = img.shape[:2]
        
        # 2. 初始化一个全黑的掩码 (单通道灰度图)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 如果没有对应的标签文件，则直接保存全黑掩码
        if not os.path.exists(label_path):
            empty_label_count += 1
        else:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                    
                # 情况A: 如果是标准的 YOLO obj bbox 格式 (长度为5)
                # 格式: class_id x_center y_center width height (所有坐标值均在0-1之间归一化)
                if len(parts) == 5:
                    class_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # 将归一化坐标转换为图像实际像素坐标
                    xc_pixel = int(xc * width)
                    yc_pixel = int(yc * height)
                    w_pixel = int(w * width)
                    h_pixel = int(h * height)
                    
                    # 计算目标框的左上角 (x1, y1) 和 右下角 (x2, y2)
                    x1 = int(xc_pixel - w_pixel / 2)
                    y1 = int(yc_pixel - h_pixel / 2)
                    x2 = int(xc_pixel + w_pixel / 2)
                    y2 = int(yc_pixel + h_pixel / 2)
                    
                    # 防止溢出画布边界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # 3. 在掩码对应位置画白色的实心矩形 (颜色为 255)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                # 情况B: 多边形掩码标注格式或其它(长度大于5)
                elif len(parts) > 5:
                    # 格式为主流的: class_id x1 y1 x2 y2 x... y...
                    # 通过求多边形顶点的最大最小 x, y 来得出该物体的目标框(bbox)
                    coords = [float(x) for x in parts[1:]]
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    
                    x1 = int(min(x_coords) * width)
                    y1 = int(min(y_coords) * height)
                    x2 = int(max(x_coords) * width)
                    y2 = int(max(y_coords) * height)
                    
                    # 防止溢出画布边界
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
        # 4. 保存掩码文件, 建议存为 .png 格式以防边缘被有损压缩
        mask_save_path = os.path.join(output_dir, f"{base_name}.png")
        cv2.imwrite(mask_save_path, mask)
        success_count += 1
        
    print("-" * 50)
    print("【制作掩码操作完成】")
    print(f"成功生成了 {success_count} 张掩码图像。")
    if empty_label_count > 0:
        print(f"注意: 其中 {empty_label_count} 张图片没有找到对应的 .txt 标签文件（已为其生成全黑背景掩码）。")
    print(f"掩码保存路径为: {output_dir}")


if __name__ == "__main__":
    # --- 在此修改你的路径 ---
    IMG_DIR = "../Datasets/coco_train2017_images"
    LABEL_DIR = "../Datasets/coco_train2017_labels_mask_txt"
    OUTPUT_DIR = "../Datasets/coco_train2017_mask_box"
    
    # # 交互式输入逻辑
    # if IMG_DIR == "/path/to/your/images":
    #     IMG_DIR = input("1. 请输入【原始图片】文件夹的绝对路径: ").strip()
    #     LABEL_DIR = input("2. 请输入【txt标签】文件夹的绝对路径: ").strip()
    #     OUTPUT_DIR = input("3. 请输入【要保存的掩码图片】目录的绝对路径: ").strip()
        
    if os.path.exists(IMG_DIR) and os.path.exists(LABEL_DIR):
         generate_masks(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
    else:
         print("错误：你提供的 图片文件夹 或 标签文件夹 不存在，请检查！")

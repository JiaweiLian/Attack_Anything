import os
import glob

def remove_extra_images(img_dir, label_dir):
    """
    删除图片文件夹中没有对应标签文件的多余图片。
    :param img_dir: 存放图片的文件夹路径
    :param label_dir: 存放标签的文件夹路径
    """
    # 1. 获取所有标签文件的基础文件名（不含扩展名.txt）
    print(f"正在读取标签所在的文件夹: {label_dir} ...")
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    if not label_files:
        print("警告：在标签文件夹中没有找到任何 .txt 文件！请检查路径是否正确。")
        return
        
    label_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in label_files])
    print(f"共找到 {len(label_basenames)} 个标签文件。")

    # 2. 遍历图片文件夹中的所有文件
    # 这里列出常见图片格式，可以根据你的实际情况增加
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    removed_count = 0
    total_images_scanned = 0

    print(f"正在扫描图片所在的文件夹: {img_dir} ...")
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(img_extensions):
            total_images_scanned += 1
            img_basename = os.path.splitext(img_name)[0]
            
            # 3. 如果图片的basename不在标签的basename集合中，则删除该图片
            if img_basename not in label_basenames:
                img_path = os.path.join(img_dir, img_name)
                try:
                    os.remove(img_path)
                    removed_count += 1
                    print(f"已删除多余图片: {img_path}")
                except Exception as e:
                    print(f"删除失败 {img_path}: {e}")

    # 4. 打印统计结果
    print("-" * 40)
    print("清理完成！")
    print(f"共扫描了 {total_images_scanned} 张图片。")
    print(f"删除了 {removed_count} 张多余图片。")
    remaining_images = total_images_scanned - removed_count
    print(f"当前剩余图片数量: {remaining_images} (标签数量: {len(label_basenames)})。")
    
    if remaining_images != len(label_basenames):
        print("提示：剩余图片数量和标签数量不一致。可能是某些标签没有对应的图片，请留意！")
    else:
        print("完美：剩余图片与标签已一一对应！")

if __name__ == "__main__":
    # 请根据实际情况修改下方这两个路径，再运行代码即可
    IMG_DIR = "../Datasets/coco_train2017_images"    # <--- 在这里填入你的图片路径
    LABEL_DIR = "../Datasets/coco_train2017_labels_mask_txt"  # <--- 在这里填入你的txt标签路径

    # if IMG_DIR == "../Datasets/coco_train2017_images" or LABEL_DIR == "../Datasets/coco_train2017_labels_mask_txt":
    #     # 如果你更喜欢从控制台直接输入，也可以取消这里注释：
    #     IMG_DIR = input("请输入图片文件夹的绝对路径: ").strip()
    #     LABEL_DIR = input("请输入标签文件夹的绝对路径: ").strip()
        
    if os.path.exists(IMG_DIR) and os.path.exists(LABEL_DIR):
        remove_extra_images(IMG_DIR, LABEL_DIR)
    else:
        print("错误：请检查提供的输入路径，图片或标签文件夹不存在！")

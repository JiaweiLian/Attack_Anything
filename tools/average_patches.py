import argparse
from PIL import Image
import numpy as np

def average_patches(patch1_path, patch2_path, output_path):
    print(f"Loading Patch 1: {patch1_path}")
    img1 = Image.open(patch1_path).convert('RGB')
    
    print(f"Loading Patch 2: {patch2_path}")
    img2 = Image.open(patch2_path).convert('RGB')

    # Ensure both patches are the same size
    if img1.size != img2.size:
        print(f"Sizes do not match. Resizing patch 2 ({img2.size}) to match patch 1 ({img1.size}).")
        img2 = img2.resize(img1.size, Image.BILINEAR)

    # Convert to numpy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    # Pixel-wise averaging (0.5 * P1 + 0.5 * P2)
    print("Performing pixel-wise averaging (0.5 * P1 + 0.5 * P2)...")
    avg_arr = (0.5 * arr1 + 0.5 * arr2).clip(0, 255).astype(np.uint8)

    # Save the output
    avg_img = Image.fromarray(avg_arr)
    avg_img.save(output_path)
    print(f"Successfully saved fused patch to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse two adversarial patches by averaging their pixels.")
    parser.add_argument("--patch1", type=str, required=True, help="Path to the first patch image.")
    parser.add_argument("--patch2", type=str, required=True, help="Path to the second patch image.")
    parser.add_argument("--output", type=str, default="fused_patch.png", help="Path to save the fused patch.")
    
    args = parser.parse_args()
    average_patches(args.patch1, args.patch2, args.output)

# python tools/average_patches.py --patch1 patches/patch_NN_response/tba_yolov5x.png --patch2 patches/patch_NN_response/tba_faster_rcnn.png --output patches/patch_NN_response/tba_yolov5x_faster_rcnn.png

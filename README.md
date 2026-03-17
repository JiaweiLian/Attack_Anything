# Attack_Anything: A Universal Framework for Adversarial Patch Attacks on Object Detectors

This repository contains the official implementation of **Attack_Anything**, a comprehensive framework for generating robust adversarial patches to attack various state-of-the-art object detection models. The framework supports powerful ensemble attacks combining different architectures (e.g. YOLO series, MMDetection models like TOOD).

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage: Training](#usage-training)
- [Usage: Evaluation](#usage-evaluation)
- [Usage: Visualization](#usage-visualization)

## Installation

We provide an `environment.yml` file to exactly reproduce the Python 3.6 conda environment used for our experiments.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JiaweiLian/Attack_Anything.git
   cd Attack_Anything
   ```

2. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate py3.6
   ```

## Repository Structure

The core codebase has been simplified to make the attack pipeline clear, with debugging and test scripts neatly organized:

- `train.py`: Core training script for optimizing adversarial patches (supports single or ensemble models).
- `evaluate.py`: Evaluation pipeline on COCO metrics (mAP validation), with physical simulation tools like viewpoint transformations.
- `visualize.py`: Renders adversarial patches onto images, supporting perspective transformations.
- `load_data.py`: Handles dataset loading, bounds calculations, and penalty losses (e.g. Total Variation, Laplacian Smoothness).
- `utils.py`: Contains common utilities for bounding box manipulation and generic functions.
- `debug_scripts/`: A dedicated folder containing various testing, debugging, and visualization exploration scripts (`test_*.py`, `fix_*.py`, etc.) for easy development.

## Usage: Training

You can train an adversarial patch targeting a single model or an ensemble of models. The framework offers multiple structural and stealth strategies such as Grid Expansion (`--grid_size`) and smoothness penalties (`--smoothness_strategy`).

**Example: Train an ensemble patch (YOLOv5x and TOOD) with TBA mode:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --ensemble yolov5x,tood \
    --attack_mode tba \
    --grid_size 8 \
    --smoothness_strategy tv_ensemble
```

## Usage: Evaluation

We evaluate the generated patches robustness across multiple scenarios, including simulating real-world physical dynamics like camera viewpoint shifts via perspective transforms.

**Example: Evaluate model robustness with a 32-degree side-angle viewpoint shift:**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --model yolov5x \
    --attack_mode tba \
    --patch_path patches/patch_NN_response/tba_yolov5x_tood_tv_laplacian.png \
    --viewpoint 32
```

## Usage: Visualization

To intuitively observe the patch's effect and its geometric transformations on standard images, use `visualize.py`. Results are automatically saved into independent folders within `visual_results/` to prevent overwriting.

**Example: Generate 100 visualizations applying a 32-degree perspective distortion:**
```bash
CUDA_VISIBLE_DEVICES=0 python visualize.py \
    --model yolov5x \
    --attack_mode tba \
    --patch_path patches/patch_NN_response/tba_yolov5x_tood_tv_laplacian.png \
    --num_images 100 \
    --viewpoint 32
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

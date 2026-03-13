from evaluate import evaluate_map
from pycocotools.coco import COCO
evaluate_map('ssd', None, attack_mode_override='clean')

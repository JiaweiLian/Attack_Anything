import evaluate
from patch_config import patch_configs

evaluate.evaluate_map('ssd', None, attack_mode_override='clean')

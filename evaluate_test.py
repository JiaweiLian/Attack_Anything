from evaluate import evaluate_map
import gc
print("Starting Evaluation Test on SSD")
evaluate_map('ssd', None, attack_mode_override='clean')

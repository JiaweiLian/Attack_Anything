import re

with open('evaluate.py', 'r') as f:
    text = f.read()

# I need to change get_mmdet_predictions to NOT replace data['img'][0] = img_tensor if we want to retain normalization! Omg!

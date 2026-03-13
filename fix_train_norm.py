import re

with open('train.py', 'r') as f:
    text = f.read()

text = text.replace("data['img'][0] = adversarial_example", "")

with open('train.py', 'w') as f:
    f.write(text)

print("train.py normalization fixed!")

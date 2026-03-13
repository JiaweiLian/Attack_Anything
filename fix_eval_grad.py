import re

with open('evaluate.py', 'r') as f:
    text = f.read()

# I also need to ensure that the tensor is correct.
# Wait, InferenceDetector internally adds another dimension, so the output could be different now? 
# let's write a small script to quickly evaluate again on 50 images to ensure evaluate.py gives 0.59 now!

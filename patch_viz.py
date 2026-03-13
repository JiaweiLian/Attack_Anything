import re

with open('visualize.py', 'r') as f:
    text = f.read()

# Fix the drawing coordinates bug where we mistakenly scaled logic.
old_draw = """                                # Coordinates are already in padded square size natively
                                draw.rectangle([bx1, by1, bx2, by2], outline="red", width=4)
                                text_str = f"{cat_name} {conf:.2f}"
                                
                                try:"""

new_draw = """                                # Coordinates from rescale=False are ALREADY in the img_size x img_size padded space.
                                draw.rectangle([bx1, by1, bx2, by2], outline="red", width=4)
                                text_str = f"{cat_name} {conf:.2f}"
                                
                                try:"""

if old_draw in text:
    print("Found draw. Already using bx1 directly.")
else:
    print("Not found old_draw.")

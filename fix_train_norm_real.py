import re

with open('train.py', 'r') as f:
    text = f.read()

find_str = """                        data['img'][0] = adversarial_example
                        output = self.model(return_loss=False, rescale=True, **data)"""

# Instead of relying on InferenceDetector pipeline (which resizes and drops gradient tracking from adversarial_example)
# We must MANUALLY normalize adversarial_example while preserving its 416x416 shape and gradient graph!
replace_str = """
                        mean = torch.tensor([123.675, 116.28, 103.53], device=adversarial_example.device).view(1, 3, 1, 1) / 255.0
                        std = torch.tensor([58.395, 57.12, 57.375], device=adversarial_example.device).view(1, 3, 1, 1) / 255.0
                        norm_tensor = (adversarial_example - mean) / std

                        data = dict(
                            img=[norm_tensor],
                            img_metas=[[dict(
                                ori_shape=(config.img_size, config.img_size, 3),
                                img_shape=(config.img_size, config.img_size, 3),
                                pad_shape=(config.img_size, config.img_size, 3),
                                scale_factor=1.0,
                                flip=False,
                            )]]
                        )
                        output = self.model(return_loss=False, rescale=False, **data)"""

if find_str in text:
    text = text.replace(find_str, replace_str)
    with open('train.py', 'w') as f:
        f.write(text)
    print("train fixed perfectly!")
else:
    print("Could not find the block to fix in train.py")

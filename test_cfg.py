from evaluate import PatchTrainer
trainer = PatchTrainer('vfnet')
print(trainer.model.cfg.img_norm_cfg)

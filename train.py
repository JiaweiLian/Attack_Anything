"""
Adversarial patch training
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Putting this command down does not work.
import PIL
import torch
from torch.optim import Adam
import load_data
from tqdm import tqdm
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
import subprocess
import argparse
import patch_config
import sys
import time
import os
import pandas as pd
import wandb
import warnings
# from ultralytics.utils.plotting import Annotator, colors, save_one_box

warnings.filterwarnings("ignore")

# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


def adv_patch_update(adv_patch_cpu, adv_patch_cpu_original, adv_patch_mask_cpu, adv_patch_reversed_mask_cpu):
    aircraft_area = torch.mul(adv_patch_cpu_original, adv_patch_mask_cpu)
    adv_patch_area = torch.mul(adv_patch_cpu, adv_patch_reversed_mask_cpu)
    adv_patch_cpu = torch.add(input=aircraft_area, alpha=1, other=adv_patch_area)
    return adv_patch_cpu


class PatchTrainer(object):
    def __init__(self, mode, attack_mode='tba', ensemble=None, grid_size=1, random_mask_prob=0.0, smoothness_strategy='tv'):
        self.mode = mode
        self.attack_mode = attack_mode
        self.ensemble = ensemble
        self.grid_size = grid_size
        self.random_mask_prob = random_mask_prob
        self.smoothness_strategy = smoothness_strategy
        self.epoch_length = 0
        
        if self.ensemble is not None:
            self.model_names = self.ensemble.split(',')
        else:
            self.model_names = [self.mode]

        self.configs = []
        self.models = []
        self.yolos = []
        self.prob_extractors = []
        self.InferenceDetectors = []

        for m in self.model_names:
            config = patch_config.patch_configs[m]()
            config.attack_mode = attack_mode
            self.configs.append(config)
            self.models.append(config.model.eval().cuda())
            is_yolo = m in ['yolov3', 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
            self.yolos.append(is_yolo)
            self.prob_extractors.append(config.prob_extractor.cuda())
            if not is_yolo:
                self.InferenceDetectors.append(config.InferenceDetector.cuda())
            else:
                self.InferenceDetectors.append(None)
                
        self.config = self.configs[0]
        self.mask_dir = self.config.box_mask_dir if attack_mode == 'bba' else self.config.mask_dir
        
        # Backward compatibility
        self.model = self.models[0]
        self.yolo = self.yolos[0]
        self.weight_and_bias = True  # Switch of wandb


    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # img_size = self.darknet_model.height
        img_size = self.config.img_size
        batch_size = self.config.batch_size
        n_epochs = 800
        # max_lab = 59 + 1  # 5l

        # # Training from existing patch
        # adv_patch_cpu = self.read_image(
        #     "patches/patch_AAAI/yolov5m_half_digital.png")  # Training from existing patch
        # adv_patch_cpu_original = self.read_image(
        #     "patches/patch_AAAI/yolov5m_half_digital.png")

        # Training from random patch
        adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu_original = self.generate_patch("gray")

        adv_patch_mask_cpu = self.read_image("./patches/patch_AAAI/grid_mask_1024.png")
        adv_patch_reversed_mask_cpu = self.read_image("./patches/patch_AAAI/grid_mask_reverse_1024.png")
        adv_patch_cpu.requires_grad_(True)  ####################################

        train_loader = torch.utils.data.DataLoader(
            COCODataset(self.config.img_dir, self.mask_dir, img_size, shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer: Adam = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        if self.attack_mode == 'bba':
            alpha = 90
        elif self.attack_mode == 'tba':
            alpha = 9

        if self.weight_and_bias:
            wandb.init(project="Adversarial-attack")
            wandb.watch_called = True  # Re-run the model without restarting the runtime, unnecessary after our next release
            wandb.watch(self.model, log="all")
            wandb.log({
                "alpha": alpha,
                "Detector": self.ensemble if self.ensemble else self.mode,
                "Patch generation": "AAAI",
                "Patch size": self.config.patch_size,
                "Batch size": self.config.batch_size,
                "Learning rate": self.config.start_learning_rate
            })

        if self.smoothness_strategy == 'tv_ensemble':
            print("Using Ensemble TV Strategy for smoothness.")
            total_variation = TotalVariation_ensemble().cuda()
        elif self.smoothness_strategy == 'laplacian':
            print("Using Laplacian Strategy for smoothness.")
            total_variation = LaplacianSmoothness().cuda()
        else:
            print("Using Vanilla TV Strategy for smoothness.")
            total_variation = TotalVariation().cuda()
            
        patch_augmentation = PatchAugmentation().cuda()
        tv_limit = torch.tensor(0.1).cuda()

        # Cache static tensors to GPU
        adv_patch_original = adv_patch_cpu_original.cuda()
        adv_patch_mask = adv_patch_mask_cpu.cuda()
        adv_patch_reversed_mask = adv_patch_reversed_mask_cpu.cuda()

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_w_and_h = 0
            for i_batch, (img_batch, mask_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                         total=self.epoch_length):
                iteration = self.epoch_length * epoch + i_batch
                # with autograd.detect_anomaly():
                if True:
                    len_batch = len(img_batch)
                    img_batch = img_batch.cuda()
                    # img_batch_show = img_batch[2]
                    mask_batch = mask_batch.cuda()
                    # mask_batch_show = mask_batch[2]
                    adv_patch = adv_patch_cpu.cuda()

                    # Mask operation
                    # adv_patch = adv_patch_update(adv_patch, adv_patch_original, adv_patch_mask, adv_patch_reversed_mask)

                    # adv_patch_show = adv_patch
                    adv_patch_unsqueezed = torch.unsqueeze(adv_patch, dim=0)
                    adv_patch_expanded = adv_patch_unsqueezed.expand(len_batch, -1, -1, -1)
                    # img_height = img_batch.shape[2]
                    # img_width = img_batch.shape[3]
                    adv_patch_resized = F.interpolate(adv_patch_expanded,
                                                      size=(self.config.img_size, self.config.img_size))
                    adv_patch_resized = patch_augmentation(adv_patch_resized)
                    # adv_patch_resized_show = adv_patch_resized[2]
                    
                    if self.random_mask_prob > 0:
                        # Adversarial Patch Dropout: Randomly mask pixels as grey (0.5), to prevent overfitting and boost transferability.
                        drop_mask = (torch.rand(len_batch, 1, self.config.img_size, self.config.img_size, device=img_batch.device) > self.random_mask_prob).float()
                        adv_patch_resized = adv_patch_resized * drop_mask + 0.5 * (1.0 - drop_mask)
                    
                    if self.attack_mode == 'bba':
                        H_img, W_img = self.config.img_size, self.config.img_size
                        strip_mask = torch.zeros_like(mask_batch)
                        strip_mask[:, :, H_img // 4 : 3 * H_img // 4, :] = 1.0
                        attack_mask = strip_mask * (1.0 - mask_batch)
                        adversarial_example = adv_patch_update(adv_patch_resized, img_batch, 1.0 - attack_mask, attack_mask)
                    elif self.attack_mode == 'ccba':
                        H_img, W_img = self.config.img_size, self.config.img_size
                        y_grid, x_grid = torch.meshgrid(torch.arange(H_img), torch.arange(W_img))
                        dist = torch.sqrt((x_grid - W_img//2)**2 + (y_grid - H_img//2)**2).to(img_batch.device)
                        ring_width = 10 # Coarser rings conforming to characteristic size in Reference 1
                        
                        # Calculate unique index for each ring
                        ring_idx = (dist / ring_width).long()
                        max_rings = int((W_img / 2) / ring_width) + 2
                        
                        radial_patch = torch.zeros_like(adv_patch_resized)
                        for r in range(max_rings):
                            mask_r = (ring_idx == r).unsqueeze(0).unsqueeze(0).float()
                            count = mask_r.sum()
                            if count > 0:
                                sum_color = (adv_patch_resized * mask_r).sum(dim=(2, 3), keepdim=True)
                                mean_color = sum_color / count
                                radial_patch += mean_color * mask_r

                        # Limit to maximum diameter = img_size (e.g., 1024)
                        valid_ring_mask = (dist <= W_img/2).float()
                        attack_mask = valid_ring_mask.unsqueeze(0).unsqueeze(0).expand_as(mask_batch) * (1.0 - mask_batch)
                        
                        adversarial_example = adv_patch_update(radial_patch, img_batch, 1.0 - attack_mask, attack_mask)
                    else: # tba
                        adversarial_example = adv_patch_update(adv_patch_resized, img_batch, mask_batch, 1 - mask_batch)
                        
                    # adversarial_example_show = adversarial_example[2]  ####################################

                    total_det_loss = 0.0
                    total_w_and_h = 0.0
                    num_models = len(self.model_names)
                    
                    for m_idx, (m_model, m_name, is_yolo, p_ext, inf_det, m_cfg) in enumerate(zip(self.models, self.model_names, self.yolos, self.prob_extractors, self.InferenceDetectors, self.configs)):
                        
                        # Ensemble Grid Partitioning
                        if num_models > 1 and self.grid_size > 1:
                            step = self.config.img_size // self.grid_size
                            y = torch.arange(self.config.img_size, device=img_batch.device)
                            x = torch.arange(self.config.img_size, device=img_batch.device)
                            gy, gx = torch.meshgrid(y, x)
                            cell_y = (gy // step).long()
                            cell_x = (gx // step).long()
                            assigned_mask = ((cell_y + cell_x) % num_models == m_idx).float().unsqueeze(0).unsqueeze(0)
                            
                            # Keep gradients only for assigned grid cells
                            m_adv_input = adversarial_example * assigned_mask + adversarial_example.detach() * (1.0 - assigned_mask)
                        else:
                            m_adv_input = adversarial_example

                        if is_yolo:  # For detectors from YOLO series
                            output = m_model(m_adv_input)
                            # print(output, output.size())  # yolov2：torch.Size([1, 100, 32, 32]) 100 = (15 + 4 + 1) * 5 = (类别数 + 坐标 + 置信度) * 锚框数
                            extracted_prob, m_w_and_h, _ = p_ext(output)
                        else:  # For detectors from MMDetection
                            adversarial_example_cpu = m_adv_input.clone()
                            adversarial_example_cpu = adversarial_example_cpu[0].detach().cpu().numpy()
                            adversarial_example_cpu = adversarial_example_cpu.reshape(1024, 1024, 3)
                            adversarial_example_cpu = adversarial_example_cpu * 255

                            data = inf_det(m_model, adversarial_example_cpu)

                            mean_val = [123.675, 116.28, 103.53]
                            std_val = [58.395, 57.12, 57.375]
                            if hasattr(m_model, 'cfg') and hasattr(m_model.cfg, 'img_norm_cfg'):
                                norm_cfg = m_model.cfg.img_norm_cfg
                                mean_val = norm_cfg.get('mean', mean_val)
                                std_val = norm_cfg.get('std', std_val)
                            
                            mean = torch.tensor(mean_val, device=m_adv_input.device).view(1, 3, 1, 1) / 255.0
                            std = torch.tensor(std_val, device=m_adv_input.device).view(1, 3, 1, 1) / 255.0
                            norm_tensor = (m_adv_input - mean) / std

                            data = dict(
                                img=[norm_tensor],
                                img_metas=[[dict(
                                    ori_shape=(m_cfg.img_size, m_cfg.img_size, 3),
                                    img_shape=(m_cfg.img_size, m_cfg.img_size, 3),
                                    pad_shape=(m_cfg.img_size, m_cfg.img_size, 3),
                                    scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                                    flip=False,
                                )]]
                            )
                            output = m_model(return_loss=False, rescale=False, **data)
                            extracted_prob, m_w_and_h = p_ext(output, m_name)
                            
                        # Accumulate
                        total_det_loss += torch.max(extracted_prob)
                        total_w_and_h += m_w_and_h
                        
                    det_loss = total_det_loss / len(self.model_names)
                    w_and_h = total_w_and_h / len(self.model_names)

                    tv = total_variation(adv_patch)
                    tv_loss = tv * alpha
                    # tv_loss = tv * alpha * 0.0
                    w_and_h /= 100.0

                    if self.attack_mode == 'bba':
                        # reference2.pdf loss: L = l_model + alpha * l_tv (no w_and_h, no tv_limit)
                        loss = det_loss + tv_loss
                    else:
                        # Your method's loss
                        loss = det_loss + torch.max(tv_loss, tv_limit) + w_and_h

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_w_and_h += w_and_h.detach().cpu().numpy() if isinstance(w_and_h, torch.Tensor) else w_and_h
                    ep_loss += loss.detach().cpu().item()

                    loss.backward()  ##########################################
                    
                    if self.attack_mode == 'bba':
                        H_patch = self.config.patch_size
                        if adv_patch_cpu.grad is not None:
                            adv_patch_cpu.grad[:, :H_patch // 4, :] = 0
                            adv_patch_cpu.grad[:, 3 * H_patch // 4:, :] = 0
                    elif self.attack_mode == 'ccba':
                        H_patch, W_patch = self.config.patch_size, self.config.patch_size
                        y_grid, x_grid = torch.meshgrid(torch.arange(H_patch), torch.arange(W_patch))
                        dist_patch = torch.sqrt((x_grid - W_patch // 2)**2 + (y_grid - H_patch // 2)**2).to(adv_patch_cpu.device)
                        valid_mask = (dist_patch <= W_patch / 2).float().unsqueeze(0).expand_as(adv_patch_cpu)
                        if adv_patch_cpu.grad is not None:
                            adv_patch_cpu.grad *= valid_mask

                    # print("adv_patch_cpu.grad sum: ", adv_patch_cpu.grad.sum().item() if adv_patch_cpu.grad is not None else None)
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    if self.attack_mode == 'bba':
                        H_patch = self.config.patch_size
                        adv_patch_cpu.data[:, :H_patch // 4, :] = 0.5
                        adv_patch_cpu.data[:, 3 * H_patch // 4:, :] = 0.5
                    elif self.attack_mode == 'ccba':
                        adv_patch_cpu.data[(valid_mask <= 0.5)] = 0.5


                    # # draw boxes on adversarial example for yolo
                    # hide_conf = False
                    # hide_labels = False
                    # names = self.model.names
                    # adversarial_example_show_ndarray = adversarial_example_show.cpu().detach().numpy().transpose(1, 2, 0)
                    # # use Annotator() need "pip install ultralytics"
                    # adversarial_example_show_ndarray = np.ascontiguousarray(adversarial_example_show_ndarray) * 255
                    # adversarial_example_show_ndarray = adversarial_example_show_ndarray.astype(np.uint8)
                    # annotator = Annotator(adversarial_example_show_ndarray, line_width=3, example=str(names))
                    # for *xyxy, conf, cls in reversed(prediction_adversarial_example2):
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    #     annotator.box_label(xyxy, label, color=colors(c, True))
                    # adversarial_example_show_ndarray = adversarial_example_show_ndarray.transpose(2, 0, 1)
                    # adversarial_example_show_ndarray = torch.from_numpy(adversarial_example_show_ndarray)

                    bt1 = time.time()
                    if self.weight_and_bias:
                        if i_batch % 1000 == 0:  ################################
                            wandb.log({
                                # "img_batch_show": wandb.Image(img_batch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "mask_batch_show": wandb.Image(mask_batch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adv_patch_show": wandb.Image(adv_patch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adv_patch_resized_show": wandb.Image(adv_patch_resized_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adversarial_example_show": wandb.Image(adversarial_example_show_ndarray, caption="patch{}".format(iteration)),  ###############################
                                # "adversarial_example_padded_show": wandb.Image(adversarial_example_padded_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adversarial_example_resized_show": wandb.Image(adversarial_example_resized_show, caption="patch{}".format(iteration)),  # Show adversarial example

                                "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                                "tv_loss": tv_loss,
                                "det_loss": det_loss,
                                "w_and_h": w_and_h,
                                "total_loss": loss,
                            })
                    # if i_batch % 50 == 0:
                    #     del len_batch, img_batch, mask_batch, adv_patch, adv_patch_original, adv_patch_mask, adv_patch_reversed_mask, adv_patch_unsqueezed, adv_patch_expanded, \
                    #         adv_patch_resized, adversarial_example, output, extracted_prob, prob_extractor, tv, w_and_h, total_variation, tv_loss, det_loss, loss
                    #     if not self.yolo:
                    #         del adversarial_example_cpu, data, InferenceDetector
                    #     torch.cuda.empty_cache()
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
            et1 = time.time()
            # print(ep_det_loss, len(train_loader))
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)
            ep_w_and_h = ep_w_and_h / len(train_loader)

            # del len_batch, img_batch, mask_batch, adv_patch, adv_patch_unsqueezed, adv_patch_expanded, \
            #     adv_patch_resized, adversarial_example, output, extracted_prob, tv, tv_loss, det_loss, loss
            # if not self.yolo:
            #     del adversarial_example_cpu, data
            # torch.cuda.empty_cache()

            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('  BOX LOSS: ', ep_w_and_h)
                print('EPOCH TIME: ', et1 - et0)
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    parser = argparse.ArgumentParser(description='Train Adversarial Patch')
    parser.add_argument('--model', type=str, default="ssd", help='Target detector (e.g. ssd, yolov5m).')
    parser.add_argument('--ensemble', type=str, default=None, help='Comma separated list of target detectors for ensemble training (e.g. yolov5s,yolov5m). Overrides --model if provided.')
    parser.add_argument('--attack_mode', type=str, default="tba", choices=['tba', 'bba', 'ccba'], help='Attack mode set in patch_config (tba, bba, or ccba).')
    parser.add_argument('--grid_size', type=int, default=1, help='Grid partitioning size for ensemble. 1 = no partition, 2 = 2x2, 4 = 4x4.')
    parser.add_argument('--random_mask_prob', type=float, default=0.0, help='Probability for random masking schedule (Adversarial Dropout). E.g. 0.3')
    parser.add_argument('--smoothness_strategy', type=str, default="tv", choices=['tv', 'tv_ensemble', 'laplacian'], help='Smoothness integration strategy to use (tv, tv_ensemble, or laplacian).')
    args = parser.parse_args()
    
    trainer = PatchTrainer(args.model, attack_mode=args.attack_mode, ensemble=args.ensemble, grid_size=args.grid_size, random_mask_prob=args.random_mask_prob, smoothness_strategy=args.smoothness_strategy)
    trainer.train()


if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES=1 python train.py --model tood --attack_mode tba
# CUDA_VISIBLE_DEVICES=0 python train.py --ensemble tood,faster_rcnn --attack_mode tba
# CUDA_VISIBLE_DEVICES=2 python train.py --model yolov5x --attack_mode tba --random_mask_prob 0.2
# CUDA_VISIBLE_DEVICES=1 python train.py --ensemble yolov5x,tood --attack_mode tba  --grid_size 8
# CUDA_VISIBLE_DEVICES=2 python train.py --ensemble yolov5x,tood --grid_size 4 --smoothness_strategy laplacian
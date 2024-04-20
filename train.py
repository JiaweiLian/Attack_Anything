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
import patch_config
import sys
import time
import os
import pandas as pd
import wandb
import warnings
from ultralytics.utils.plotting import Annotator, colors, save_one_box

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
    def __init__(self, mode):
        self.mode = mode
        self.epoch_length = 0
        self.config = patch_config.patch_configs[mode]()
        self.model = self.config.model.eval().cuda()
        self.weight_and_bias = True  # Switch of wandb
        self.yolo = True
        if self.mode not in ['yolov3', 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
            self.yolo = False


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

        adv_patch_mask_cpu = self.read_image("/home/jiawei/adversarial-yolo/patches/patch_AAAI/grid_mask_1024.png")
        adv_patch_reversed_mask_cpu = self.read_image("/home/jiawei/adversarial-yolo/patches/patch_AAAI/grid_mask_reverse_1024.png")
        adv_patch_cpu.requires_grad_(True)  ####################################

        train_loader = torch.utils.data.DataLoader(
            COCODataset(self.config.img_dir, self.config.mask_dir, img_size, shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer: Adam = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        alpha = 9

        if self.weight_and_bias:
            wandb.init(project="Adversarial-attack")
            wandb.watch_called = True  # Re-run the model without restarting the runtime, unnecessary after our next release
            wandb.watch(self.model, log="all")
            wandb.log({
                "alpha": alpha,
                "Detector": self.mode,
                "Patch generation": "AAAI",
                "Patch size": self.config.patch_size,
                "Batch size": self.config.batch_size,
                "Learning rate": self.config.start_learning_rate
            })

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_w_and_h = 0
            for i_batch, (img_batch, mask_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                         total=self.epoch_length):
                iteration = self.epoch_length * epoch + i_batch
                with autograd.detect_anomaly():
                    if self.yolo:
                        prob_extractor = self.config.prob_extractor.cuda()  # For detectors from YOLO series
                    else:
                        prob_extractor = self.config.prob_extractor.cuda()
                        InferenceDetector = self.config.InferenceDetector.cuda()  # For detectors from MMDetection
                    # total_variation = TotalVariation_ensemble().cuda()
                    total_variation = TotalVariation().cuda()
                    patch_augmentation = PatchAugmentation().cuda()

                    len_batch = len(img_batch)
                    img_batch = img_batch.cuda()
                    # img_batch_show = img_batch[2]
                    mask_batch = mask_batch.cuda()
                    # mask_batch_show = mask_batch[2]
                    adv_patch = adv_patch_cpu.cuda()

                    # Mask operation
                    adv_patch_original = adv_patch_cpu_original.cuda()
                    adv_patch_mask = adv_patch_mask_cpu.cuda()
                    adv_patch_reversed_mask = adv_patch_reversed_mask_cpu.cuda()
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
                    adversarial_example = adv_patch_update(adv_patch_resized, img_batch, mask_batch, 1 - mask_batch)
                    adversarial_example_show = adversarial_example[2]  ####################################

                    if self.yolo:  # For detectors from YOLO series
                        output = self.model(adversarial_example)
                        # print(output, output.size())  # yolov2：torch.Size([1, 100, 32, 32]) 100 = (15 + 4 + 1) * 5 = (类别数 + 坐标 + 置信度) * 锚框数
                        extracted_prob, w_and_h, prediction_adversarial_example2 = prob_extractor(output)
                        # print(max_prob, max_prob.size())  # tensor([0.81629], device='cuda:0', grad_fn=<MaxBackward0>) torch.Size([1])
                    else:  # For detectors from MMDetection
                        adversarial_example_cpu = adversarial_example.clone()
                        adversarial_example_cpu = adversarial_example_cpu[0].detach().cpu().numpy()
                        adversarial_example_cpu = adversarial_example_cpu.reshape(1024, 1024, 3)
                        adversarial_example_cpu = adversarial_example_cpu * 255

                        data = InferenceDetector(self.model, adversarial_example_cpu)
                        data['img'][0] = adversarial_example
                        output = self.model(return_loss=False, rescale=True, **data)
                        # print(output)
                        extracted_prob, w_and_h = prob_extractor(output, self.mode)

                    tv = total_variation(adv_patch)
                    tv_loss = tv * alpha
                    # tv_loss = tv * alpha * 0.0
                    det_loss = torch.max(extracted_prob)
                    w_and_h /= 100.0
                    loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + w_and_h
                    # loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_w_and_h += w_and_h.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()  ##########################################
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    # # draw boxes on adversarial example (yolo)
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
                        if i_batch % 50 == 0:  ################################
                            wandb.log({
                                # "img_batch_show": wandb.Image(img_batch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "mask_batch_show": wandb.Image(mask_batch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adv_patch_show": wandb.Image(adv_patch_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adv_patch_resized_show": wandb.Image(adv_patch_resized_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                "adversarial_example_show": wandb.Image(adversarial_example_show_ndarray, caption="patch{}".format(iteration)),  ###############################
                                # "adversarial_example_padded_show": wandb.Image(adversarial_example_padded_show, caption="patch{}".format(iteration)),  # Show adversarial example
                                # "adversarial_example_resized_show": wandb.Image(adversarial_example_resized_show, caption="patch{}".format(iteration)),  # Show adversarial example

                                "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                                "tv_loss": tv_loss,
                                "det_loss": det_loss,
                                "w_and_h": w_and_h,
                                "total_loss": loss,
                            })
                    if i_batch % 50 == 0:
                        del len_batch, img_batch, mask_batch, adv_patch, adv_patch_original, adv_patch_mask, adv_patch_reversed_mask, adv_patch_unsqueezed, adv_patch_expanded, \
                            adv_patch_resized, adversarial_example, output, extracted_prob, prob_extractor, tv, w_and_h, total_variation, tv_loss, det_loss, loss
                        if not self.yolo:
                            del adversarial_example_cpu, data, InferenceDetector
                        torch.cuda.empty_cache()
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
    trainer = PatchTrainer('ssd')
    trainer.train()


if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES=4 python train_patch_AAAI.py

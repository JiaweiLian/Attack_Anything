import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_region_boxes, nms
from median_pool import MedianPool2d
from utils_yolov5.general import non_max_suppression

print('starting test read')
im = Image.open('/home/jiawei/adversarial-yolo/data/horse.jpg').convert('RGB')
print('img read!')


class MaxProbExtractor_yolov2(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target):
        super(MaxProbExtractor_yolov2, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target

    def forward(self, YOLOoutput):
        # get values neccesary for transformation
        if YOLOoutput.dim() == 3:
            YOLOoutput = YOLOoutput.unsqueeze(0)
        batch = YOLOoutput.size(0)
        assert (YOLOoutput.size(1) == (5 + self.num_cls) * 5)
        h = YOLOoutput.size(2)
        w = YOLOoutput.size(3)
        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = YOLOoutput.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]
        # confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
        # print(confs_if_object,len(confs_if_object[0]))
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
        # print(max_conf)

        return max_conf


class MeanProbExtractor_yolov2(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target, conf_thres, iou_thres, max_det, model):
        super(MeanProbExtractor_yolov2, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.model = model

    def forward(self, output):
        output = \
        get_region_boxes(output, self.conf_thres, self.model.num_classes, self.model.anchors, self.model.num_anchors)[0]
        output = nms(output, self.iou_thres)
        conf_addition = torch.tensor(0.0).cuda()
        len_output = len(output)
        if len(output) > 0:
            for i in range(len_output):
                output_i = output[i]
                conf_addition += output_i[4]  # The mean value of confs
            mean_conf = conf_addition / len_output
        else:
            mean_conf = torch.tensor(float(0))
        return mean_conf


class MaxProbExtractor_yolov5(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target):
        super(MaxProbExtractor_yolov5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target

    def forward(self, YOLOoutput):  # YOLOoutput: torch.Size([batch, 64512, 20])
        output = YOLOoutput.transpose(1, 2).contiguous()  # [batch, 20, 64512]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 64512]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 15, 64512]
        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)
        # we only care for probabilities of the class of interest (Plane)
        confs_for_class = normal_confs[:, self.cls_id, :]
        # confs_if_object = output_objectness  # confs_for_class * output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.loss_target(output_objectness, confs_for_class)
        # print(confs_if_object, len(confs_for_class[0]))
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)
        # print(max_conf)

        return max_conf


class MeanProbExtractor_yolov5(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, loss_target, conf_thres, iou_thres, max_det):
        super(MeanProbExtractor_yolov5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = loss_target
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, YOLOoutput):  # YOLOoutput: torch.Size([batch, 64512, 20])
        output = non_max_suppression(YOLOoutput, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=self.max_det)
        # print(output)
        # label = output[0][:, -1]  # Extract label column, e.g. tensor([ 0.,  0.,  0.,  0., 14.])
        # index = label == 0.0  # Labels are plane, e.g. tensor([ True,  True,  True,  True, False])
        # output = output[0][index]  # Predictions of plane
        # print(output)
        conf_addition = torch.tensor(0.0).cuda()
        w_and_h = torch.tensor(0.0).cuda()
        len_output = len(output)
        if len_output > 0:
            for i in range(len_output):
                output_i = output[i]
                len_output_i = len(output_i)
                if len_output_i > 0:
                    mean_conf = torch.max(output_i[:, 4])  # mean conf or max conf
                    scores = output_i[:, 4]
                    w_and_h += torch.mean(torch.abs(output_i[:, 0:2] - output_i[:, 2:4]) * scores[:, None])
                else:  # Without objects
                    mean_conf = torch.tensor(float(0))
                conf_addition += mean_conf
            conf_addition /= len_output
            w_and_h /= len_output
        return conf_addition, w_and_h, output[2]  # new loss


class MeanProbExtractor_mmdetection(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self):
        super(MeanProbExtractor_mmdetection, self).__init__()
        self.confidence_threshold = 0.4
        self. mask = []

    def forward(self, output, mode):  # YOLOoutput: torch.Size([batch, 64512, 20])
        detected_object_class_number = torch.tensor(float(0)).cuda()
        mean_class_confidence_addition = torch.tensor(float(0)).cuda()
        w_and_h = torch.tensor(0.0).cuda()
        if mode in ['ssd', 'faster_rcnn', 'cascade_rcnn', 'retinanet', 'foveabox', 'free_anchor',
                         'fsaf', 'reppoints', 'deformable_detr', 'tood', 'atss', 'vfnet']:
            output = output[0]
        elif mode in ['swin', 'mask_rcnn']:
            output = output[0][0]
        for class_i in output:
            if len(class_i) > 0:
                # mask = class_i[:, 4] >= self.confidence_threshold
                # class_i = class_i[mask]
                # if len(class_i) > 0:
                detected_object_class_number += 1.0
                mean_class_confidence_addition += class_i[:, 4].mean()
                scores = class_i[:, 4]
                w_and_h += torch.mean(torch.abs(class_i[:, 0:2] - class_i[:, 2:4]) * scores[:, None])
        if detected_object_class_number > 0:
            mean_class_confidence_addition /= detected_object_class_number
            w_and_h /= detected_object_class_number
        return mean_class_confidence_addition, w_and_h


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001),
                            0)  # torch.sum(x, 0): calculate the sum of the 0th dimension elements in the tensor x
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)  # torch.numel(): count the total number of elements in a tensor


class TotalVariation_ensemble(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation_ensemble, self).__init__()
        self.junction_columns = torch.tensor([254, 510, 766]).cuda()
        self.span_columns = 20

    def forward(self, adv_patch):  # start refactor
        adv_patch_original = adv_patch.clone()
        adv_patch = torch.flip(adv_patch, dims=[1])
        adv_patch = torch.flip(adv_patch, dims=[2])
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, :-1] - adv_patch[:, :, 1:] + 0.000001),
                            0)  # torch.sum(x, 0): calculate the sum of the 0th dimension elements in the tensor x
        for column in self.junction_columns:
            adjacent_columns = torch.arange(column - self.span_columns, column + self.span_columns + 1).cuda()
            distances = torch.abs(adjacent_columns - column).cuda()
            coefficients = self.span_columns / (distances + 1e-6)
            coefficients[self.span_columns] = 1.0
            tvcomp1[:, adjacent_columns] *= coefficients
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)

        tvcomp2 = torch.sum(torch.abs(adv_patch[:, :-1, :] - adv_patch[:, 1:, :] + 0.000001), 0).transpose(0, 1)
        junction_rows = self.junction_columns
        for row in junction_rows.cuda():
            adjacent_rows = torch.arange(row - self.span_columns, row + self.span_columns + 1).cuda()
            distances = torch.abs(adjacent_rows - row).cuda()
            coefficients = self.span_columns / (distances + 1e-6)
            coefficients[self.span_columns] = 1.0
            tvcomp2[:, adjacent_rows] *= coefficients
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv1 = tvcomp1 + tvcomp2

        tvcomp1 = torch.sum(torch.abs(adv_patch_original[:, :, :-1] - adv_patch_original[:, :, 1:] + 0.000001),
                            0)  # torch.sum(x, 0): calculate the sum of the 0th dimension elements in the tensor x
        for column in self.junction_columns:
            adjacent_columns = torch.arange(column - self.span_columns, column + self.span_columns + 1).cuda()
            distances = torch.abs(adjacent_columns - column).cuda()
            coefficients = self.span_columns / (distances + 1e-6)
            coefficients[self.span_columns] = 1.0
            tvcomp1[:, adjacent_columns] *= coefficients
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)

        tvcomp2 = torch.sum(torch.abs(adv_patch_original[:, :-1, :] - adv_patch_original[:, 1:, :] + 0.000001),
                            0).transpose(0, 1)
        junction_rows = self.junction_columns
        for row in junction_rows.cuda():
            adjacent_rows = torch.arange(row - self.span_columns, row + self.span_columns + 1).cuda()
            distances = torch.abs(adjacent_rows - row).cuda()
            coefficients = self.span_columns / (distances + 1e-6)
            coefficients[self.span_columns] = 1.0
            tvcomp2[:, adjacent_rows] *= coefficients
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv2 = tvcomp1 + tvcomp2

        tv = (tv2 + tv1) / 2.0

        return tv / torch.numel(adv_patch)  # torch.numel(): count the total number of elements in a tensor


class TricolorCamouflage(nn.Module):
    """TricolorCamouflage: calculates the camouflage loss of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __int__(self):
        super(TricolorCamouflage, self).__init__()

    def forward(self, adv_patch):
        # Earthy color (194 142 0), light green (0 220 0), and dark green (0 139 0) initialization.
        earthy = torch.zeros_like(adv_patch, dtype=torch.float)
        earthy[0] = 194 / 255
        earthy[1] = 142 / 255
        earthy[2] = 0
        light_green = torch.zeros_like(adv_patch, dtype=torch.float)
        light_green[1] = 220 / 255
        dark_green = torch.zeros_like(adv_patch, dtype=torch.float)
        dark_green[1] = 139 / 255

        # Calculate the errors of adv_patch and 3 color, respectively.
        error_earthy = abs(adv_patch - earthy)
        error_light_green = abs(adv_patch - light_green)
        error_dark_green = abs(adv_patch - dark_green)

        # Select the smallest error.
        error = torch.minimum(error_earthy, error_light_green)
        error = torch.minimum(error, error_dark_green)

        return error.mean()


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y
        #########################################################################################
        # Put the patch in the upper of the target
        target_y = target_y - 0.4 * (target_size / current_patch_size / 3 * 5)
        #########################################################################################
        # Adjust the patch size on the object. The bigger the scale, the bigger the patch size.
        target_size /= 2.0

        ############################################################################################
        scale = target_size / current_patch_size * 4  # patch outside targets
        # scale = target_size / current_patch_size  # patch on targets
        ############################################################################################

        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)

        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        # img = adv_batch_t * msk_batch_t
        # img = img[0, 0, :, :, :].detach().cpu()
        # img = transforms.ToPILImage()(img)
        # img.save("adv_batch_t.jpg")
        # img.show()
        # exit()

        return adv_batch_t * msk_batch_t


class PatchAugmentation(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchAugmentation, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

    def forward(self, adv_patch):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        # adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        # pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        # adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        # adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = adv_patch.size(0)

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_patch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_patch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        return adv_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)  # Make the labels of different images have same size. batch_size=1时不需要
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        # print(pad_size)
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        # print(padded_lab.size())
        return padded_lab


class COCODataset(Dataset):
    """

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, mask_dir, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_png_masks = len(fnmatch.filter(os.listdir(mask_dir), '*.png'))
        n_jpg_masks = len(fnmatch.filter(os.listdir(mask_dir), '*.jpg'))
        n_masks = n_png_masks + n_jpg_masks
        assert n_images == n_masks, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgsize = imgsize
        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir)
        self.img_names.sort()
        self.mask_names.sort()
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.mask_paths = []
        for img_name in self.img_names:
            self.mask_paths.append(os.path.join(self.mask_dir, img_name).replace(".png", "jpg"))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        image, mask = self.pad_and_scale(image, mask)
        transform = transforms.ToTensor()
        image = transform(image)
        mask = transform(mask)
        return image, mask

    def pad_and_scale(self, img, mask):
        """

        Args:
            img:
            mask:
        Returns:

        """
        assert img.size == mask.size
        w, h = img.size
        if w == h:
            padded_img = img
            padded_mask = mask
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                padded_mask = Image.new('RGB', (h, h), color=(0, 0, 0))
                padded_mask.paste(mask, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                padded_mask = Image.new('RGB', (w, w), color=(0, 0, 0))
                padded_mask.paste(mask, (0, int(padding)))
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)  # choose here
        padded_mask = resize(padded_mask)  # choose here
        return padded_img, padded_mask

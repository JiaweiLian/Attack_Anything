from torch import optim
from load_data import MaxProbExtractor_yolov2, MaxProbExtractor_yolov5, MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2, MeanProbExtractor_mmdetection
from mmdet.apis.inference import InferenceDetector
from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils_yolov5.torch_utils import select_device, time_sync
import os
from mmdet.apis import (async_inference_detector, InferenceDetector,
                        init_detector, show_result_pyplot)


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # self.img_dir = "/home/jiawei/datasets/coco/train2017_objects"
        # self.mask_dir = "/home/jiawei/datasets/coco/train2017_mask"
        # self.img_dir = "/home/jiawei/datasets/AA/car/clean360"
        # self.mask_dir = "/home/jiawei/datasets/AA/car/mask360"
        self.img_dir = "/home/jiawei/datasets/AA/mix/clean360"
        self.mask_dir = "/home/jiawei/datasets/AA/mix/mask360"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 1024
        self.start_learning_rate = 0.03
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.batch_size = 8
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.15  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        # self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls
        self.loss_target = lambda obj, cls: obj


class yolov3(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        self.patch_name = 'ObjectOnlyPaper'
        self.batch_size = 4
        self.weights_yolov3 = "/home/jiawei/yolov3-master/yolov3.pt"
        self.data = '/home/jiawei/yolov3-master/data/coco.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend_yolov3(self.weights_yolov3,
                                               device=self.device,
                                               dnn=False, ).eval()
        self.prob_extractor = MeanProbExtractor_yolov5(0, 80, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5n(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.batch_size = 16
        self.weights_yolov5 = "/mnt/lianjiawei/yolov5/yolov5n.pt"
        self.data = '/mnt/lianjiawei/yolov5/data/coco.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 80, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5s(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5s/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5m(BaseConfig):  # Trained on COCO
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.batch_size = 8
        self.weights_yolov5 = "/mnt/lianjiawei/yolov5/yolov5m.pt"
        self.data = '/mnt/lianjiawei/yolov5/data/coco.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 80, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5l(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.batch_size = 6
        self.weights_yolov5 = "/mnt/lianjiawei/yolov5/yolov5l.pt"
        self.data = '/mnt/lianjiawei/yolov5/data/coco.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        self.prob_extractor = MeanProbExtractor_yolov5(0, 80, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5x(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.patch_name = 'ObjectOnlyPaper'
        self.weights_yolov5 = "/home/jiawei/models/detectors/yolov5x.pt"
        self.data = '/home/jiawei/yolov5/data/coco.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        self.prob_extractor = MeanProbExtractor_yolov5(0, 80, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class faster_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/faster_rcnn_r50_fpn_2x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class ssd(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 10
        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py'
        self.checkpoint_file = "/home/jiawei/models/detectors/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth"  # For COCO
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class swin(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
        self.checkpoint_file = "/home/jiawei/models/object_detectors/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth"  # COCO
        self.batch_size = 6
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class cascade_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/cascade_rcnn_r50_fpn_20e_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class retinanet(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = "configs/retinanet/retinanet_r50_fpn_2x_coco.py"
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/retinanet_r50_fpn_2x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class mask_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_fp16_1x_coco.py'
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/mask_rcnn_r50_fpn_fp16_1x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class foveabox(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py'
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/fovea_r50_fpn_4x4_2x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class free_anchor(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = "configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py"
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/retinanet_free_anchor_r50_fpn_1x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class fsaf(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = "configs/fsaf/fsaf_r50_fpn_1x_coco.py"
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/fsaf_r50_fpn_1x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class reppoints(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py'
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/reppoints_moment_r50_fpn_1x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class tood(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = "configs/tood/tood_r50_fpn_mstrain_2x_coco.py"
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/tood_r50_fpn_mstrain_2x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class atss(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 8
        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = "configs/atss/atss_r50_fpn_1x_coco.py"
        self.checkpoint_file = "/mnt/lianjiawei/mmdetection-master/models/atss_r50_fpn_1x_coco.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


class vfnet(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/vfnet/vfnet_r50_fpn_1x_coco.py'
        self.checkpoint_file = "/data/lianjiawei/mmdetection-master/work_dirs/vfnet_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()
        self.prob_extractor = MeanProbExtractor_mmdetection()


patch_configs = {
    "base": BaseConfig,
    "yolov3": yolov3,
    "yolov5n": yolov5n,
    "yolov5s": yolov5s,
    "yolov5m": yolov5m,
    "yolov5l": yolov5l,
    "yolov5x": yolov5x,
    "faster_rcnn": faster_rcnn,
    "swin": swin,
    "ssd": ssd,
    "cascade_rcnn": cascade_rcnn,
    "retinanet": retinanet,
    "mask_rcnn": mask_rcnn,
    "foveabox": foveabox,  # anchor-free
    "free_anchor": free_anchor,
    "fsaf": fsaf,  # anchor-free, single-shot
    "reppoints": reppoints,
    "tood": tood,  # one-stage
    "atss": atss,
    "vfnet": vfnet,
}

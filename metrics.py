import os
import glob
import pandas as pd
from utils import AnnotationDF
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

class COCO_annotation_loader():
    def __init__(self, ann_dir):
        self.ann_dir = ann_dir
        self.files = []
        self.filenames = []
        self._search_files()

    def _search_files(self):
        self.files = sorted(glob.glob(self.ann_dir + '/*' + '.txt')) 
        self.filenames = [os.path.split(x)[-1].split('.')[0] for x in self.files]

    def __len__(self):
        return len(self.files)
    
    # Читает файл и возвращает np array
    def __read_annotation(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        annotation = np.zeros((len(lines), 5), dtype=np.float32) # Class, x,y,w,h
        for index, line in enumerate(lines):
            class_index = int(line.split(' ')[0])
            bbox = np.array([float(x) for x in line.rstrip('\n').split(' ')[1:]])
            annotation[index, 0] = class_index
            annotation[index, 1:] = bbox
        return annotation
    
    # Возвращает numpy array по индексу
    def get_annotation_by_name(self, name):
        try:
            idx = self.filenames.index(name)
        except ValueError:
            print('not in the list')
            return None
        ann_path = self.files[idx]
        return self.__read_annotation(ann_path)


    # Возвращает numpy array по индексу
    def __getitem__(self, idx):
        ann_path = self.files[idx]
        filename =  self.filenames[idx]
        annotation = self.__read_annotation(ann_path)
        return filename, annotation  
    


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    ap = np.zeros((nc, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
    return ap, unique_classes.astype(int)


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.
    Source: https://github.com/ultralytics/yolov5
    """
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Source: https://github.com/ultralytics/yolov5

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.
    Source: https://github.com/ultralytics/yolov5

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)

def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    Source: https://github.com/ultralytics/yolov5
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def count_metrics(det_loader, ann_loader):
    stats, ap, ap_class = [], [], [], []
    single_cls=False

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    for image_name, labels in tqdm(ann_loader):
    # Metrics
        #get preds
        labels = torch.from_numpy(labels)
        preds = torch.from_numpy(det_loader.get_annotation_by_name(image_name))

        nl, npr = labels.shape[0], preds.shape[0]  # number of labels, predictions
        correct = torch.zeros(npr, niou, dtype=torch.bool)

        if npr == 0:
            if nl:
                stats.append((correct, *torch.zeros((2, 0)), labels[:, 0]))
            continue

        # Predictions
        if single_cls:
            preds[:, 5] = 0
        

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            labelsn = torch.cat((labels[:, 0:1], tbox), 1) # labels in (class, x1,y1,x2,y2) format
            pbox = xywh2xyxy(preds[:, 0:4])  # pred boxes
            predsn = torch.cat((pbox, preds[:, 4:]), 1) # preds in (x1,y1,x2,y2, confidence, class) format
            correct = process_batch(predsn, labelsn, iouv)

        stats.append((correct, preds[:, 4], preds[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    
    # Compute metrics
    numpy_stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(numpy_stats) and numpy_stats[0].any():
        ap, ap_class = ap_per_class(*numpy_stats)

    return ap, ap_class
import os
import shutil
import numpy as np
import cv2
from utils import vis_tools
from utils import utils
from utils import math
from utils import transform
from config.config import cfg


def parse_bev_predmap(predmap, anchors):
    xmap = np.tile(np.array(range(cfg.BEV.OUTPUT_Y))[:, np.newaxis], [1, cfg.BEV.OUTPUT_X])
    ymap = np.tile(np.array(range(cfg.BEV.OUTPUT_X))[np.newaxis, :], [cfg.BEV.OUTPUT_Y, 1])
    xy_grid = np.stack((xmap,ymap), axis=-1)
    predmap = np.concatenate((predmap, xy_grid), axis=-1)
    preds = predmap[math.sigmoid(predmap[..., 0])>0.6]
    objness = math.sigmoid(preds[..., 0])[..., np.newaxis]
    clsness = math.sigmoid(preds[..., 1:cfg.CONTFUSE.CLASSES_NUM+1])
    box = preds[..., cfg.CONTFUSE.CLASSES_NUM+1:-2].reshape(-1, cfg.CONTFUSE.CLASSES_NUM, cfg.BEV.BBOX_DIM)
    prob = clsness * objness
    cls_max_prob = np.max(prob, axis=-1)
    cls_idx = np.argmax(prob, axis=-1)
    box = box[np.arange(box.shape[0]), cls_idx]
    xx = preds[..., -2] - box[..., 0] * anchors[cls_idx, 3]
    yy = preds[..., -1] - box[..., 1] * anchors[cls_idx, 4]
    x = cfg.BEV.X_MAX - xx * cfg.BEV.X_RESOLUTION * cfg.BEV.STRIDE
    y = cfg.BEV.Y_MAX - yy * cfg.BEV.Y_RESOLUTION * cfg.BEV.STRIDE
    hwl = box[..., 2:5] * anchors[cls_idx][..., :3]
    theta = np.arctan2(np.sin(box[..., 5]), np.cos(box[..., 5]))
    result = np.stack([cls_idx, cls_max_prob, x, y, hwl[..., 0], hwl[..., 1], hwl[..., 2], theta], axis=-1)
    return result[cls_max_prob>0.6]


def parse_img_predmap(predmap, anchors):
    anchor_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, anchors.shape[0], anchors.shape[1]]
    anchors = np.broadcast_to(np.array(anchors), anchor_shape)
    h = np.tile(np.array(range(cfg.IMAGE.OUTPUT_H))[:, np.newaxis], [1, cfg.IMAGE.OUTPUT_W])
    w = np.tile(np.array(range(cfg.IMAGE.OUTPUT_W))[np.newaxis, :], [cfg.IMAGE.OUTPUT_H, 1])
    hw_grid = np.stack((h, w), axis=-1)
    hw_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.ANCHORS_NUM, 2]
    hw_grid = np.tile(hw_grid, cfg.IMAGE.ANCHORS_NUM).reshape(hw_shape) 
    box_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.ANCHORS_NUM, cfg.CONTFUSE.CLASSES_NUM+cfg.IMAGE.BBOX_DIM+1]
    predmap = predmap.reshape(box_shape)
    predmap = np.concatenate((predmap, hw_grid, anchors), axis=-1)
    preds = predmap[math.sigmoid(predmap[..., 0])>0.5]
    objness = math.sigmoid(preds[..., 0])[..., np.newaxis]
    clsness = math.sigmoid(preds[..., 1:cfg.CONTFUSE.CLASSES_NUM+1])
    box = preds[..., cfg.CONTFUSE.CLASSES_NUM+1:]
    prob = objness * clsness
    cls_max_prob = np.max(prob, axis=-1)
    cls_idx = np.argmax(prob, axis=-1)
    x = (box[:, 0] + box[:, -4]) * cfg.IMAGE.STRIDE / cfg.IMAGE.H_SCALE_RATIO 
    y = (box[:, 1] + box[:, -3]) * cfg.IMAGE.STRIDE / cfg.IMAGE.W_SCALE_RATIO
    h = box[:, 2] / cfg.IMAGE.H_SCALE_RATIO * box[:, -2]
    w = box[:, 3] / cfg.IMAGE.W_SCALE_RATIO * box[:, -1]
    left = y - w / 2
    top = x - h / 2
    right = y + w / 2
    bottom = x + h / 2
    result = np.stack([cls_idx, cls_max_prob, left, top, right, bottom], axis=-1)
    return result[cls_max_prob>0.5]


def img_nms(bboxes, iou_threshold, sigma=0.3, method='nms'):

    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious

    classes_in_img = list(set(bboxes[:, 0]))
    best_bboxes = []
    for cls_type in classes_in_img:
        cls_mask = (bboxes[:, 0] == cls_type)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes):
            max_ind = np.argmax(cls_bboxes[:, 1])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind+1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, 2:], cls_bboxes[:, 2:])
            weight = np.ones((len(iou),), dtype=np.float32)
            assert method in ['nms', 'soft-nms']
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 1] = cls_bboxes[:, 1] * weight
            score_mask = cls_bboxes[:, 1] > 0.
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


def bev_nms(bboxes, thresholds):

    def is_close_center(bbox1, bbox2, threshold):
        xy = bbox2[..., 2:4] - bbox1[..., 2:4]
        distance = np.sqrt(xy[:,0]*xy[:,0]+ xy[:,1]*xy[:,1])
        return distance < threshold

    def merge_obj_bboxes(bboxes, cls_type):
        new_box = np.zeros(bboxes.shape[-1])
        new_box[0] = cls_type
        # print(bboxes[..., -1])
        new_box[1] = np.mean(bboxes[..., 1])
        new_box[2:] = np.sum(bboxes[..., 2:]*bboxes[..., 1][..., np.newaxis], axis=0)
        sum_of_prob = np.sum(bboxes[..., 1])
        new_box[2:] = new_box[2:] / sum_of_prob
        area = new_box[5] * new_box[6]        
        if sum_of_prob / area < 1:
            return []
        return new_box
    
    classes_in_bev = list(set(bboxes[:, 0]))
    best_bboxes = []
    for cls_type in classes_in_bev:
        cls_mask = (bboxes[:, 0] == cls_type)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes):
            max_ind = np.argmax(cls_bboxes[:, 1])
            sample_bbox = cls_bboxes[max_ind]
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind+1: ]])
            distance_mask = is_close_center(sample_bbox, cls_bboxes, thresholds[int(cls_type)])
            obj_bboxes = cls_bboxes[distance_mask]
            merged_bbox = merge_obj_bboxes(obj_bboxes, cls_type)
            if len(merged_bbox):
                best_bboxes.append(merged_bbox)
            cls_bboxes = cls_bboxes[np.logical_not(distance_mask)]
    return best_bboxes


def save_lidar_results(bev_bboxes, tr, frame_id, save_path):
    bev_file = os.path.join(save_path, frame_id+'.txt')
    f1 = open(bev_file,'w')
    for box in bev_bboxes:
        pred_cls = cfg.CONTFUSE.CLASSES_LIST[int(box[0])]
        location = transform.location_lidar2cam(box[2:5], tr)
        ry = transform.rz_to_ry(box[-1])
        line = pred_cls + " -1.0 -1.0 -10.0 -1.0 -1.0 -1.0 -1.0 "
        line += "{:.2f} {:.2f} {:.2f} ".format(box[3], box[4], box[5])
        line += "{:.2f} {:.2f} {:.2f} ".format(location[0], location[1], location[2])
        line += "{:.2f} {:.2f}\n".format(box[1], ry)
        f1.write(line)
    f1.close()
    return


def save_image_results(img_bboxes, frame_id, save_path):
    img_file = os.path.join(save_path, frame_id+'.txt')
    f1 = open(img_file,'w')
    for box in img_bboxes:
        pred_cls = cfg.CONTFUSE.CLASSES_LIST[int(box[0])]
        line = pred_cls + " -1.0 -1.0 -10.0 "
        line += "{:.2f} {:.2f} {:.2f} {:.2f} ".format(box[2], box[3], box[4], box[5])
        line += "-10.0 -10.0 -10.0 -10.0 -10.0 -10.0 -10.0"
        line += "{:.2f}\n".format(box[1])
        f1.write(line)
    f1.close()
    return


import sys
sys.path.append("../")
import os
import cv2
from data import dataset
from utils import utils
from utils import vis_tools
from tqdm import tqdm
from config.config import cfg
from data import postprocess
from data import preprocess
from data import loader
import numpy as np



def get_idx(array):
	idx_tuple = np.where(array==1)
	u, idx = np.unique(idx_tuple[0], return_index = True)
	return u, idx_tuple[1][idx]


def parse_bevlabel(bevlabel, anchors):
	xmap = np.tile(np.array(range(cfg.BEV.OUTPUT_Y))[:, np.newaxis], [1, cfg.BEV.OUTPUT_X])
	ymap = np.tile(np.array(range(cfg.BEV.OUTPUT_X))[np.newaxis, :], [cfg.BEV.OUTPUT_Y, 1])
	xy_grid = np.stack((xmap,ymap), axis=-1)
	bevlabel = np.concatenate((bevlabel, xy_grid), axis=-1)
	labels = bevlabel[bevlabel[..., 0]==1]
	cls_type = labels[..., 1].astype(np.int32)
	prob = np.ones(cls_type.shape[0], dtype=np.float32)
	box = labels[..., 3:-2].reshape(-1, cfg.CONTFUSE.CLASSES_NUM, cfg.BEV.BBOX_DIM)
	box = box[np.arange(box.shape[0]), cls_type]
	xx = labels[..., -2] - box[..., 0] * anchors[cls_type,3] 
	yy = labels[..., -1] - box[..., 1] * anchors[cls_type,4]
	x = cfg.BEV.X_MAX - xx * cfg.BEV.X_RESOLUTION * cfg.BEV.STRIDE
	y = cfg.BEV.Y_MAX - yy * cfg.BEV.Y_RESOLUTION * cfg.BEV.STRIDE
	hwl = box[..., 2:5] * anchors[cls_type, :3]
	theta = np.arctan2(np.sin(box[..., 5]), np.cos(box[..., 5]))
	return np.stack([cls_type, prob, x, y, hwl[..., 0], hwl[..., 1], hwl[..., 2], theta], axis=-1)


def parse_imglabel(imglabel, anchors):
	anchor_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, anchors.shape[0], anchors.shape[1]]
	anchors = np.broadcast_to(np.array(anchors), anchor_shape)
	h = np.tile(np.array(range(cfg.IMAGE.OUTPUT_H))[:, np.newaxis], [1, cfg.IMAGE.OUTPUT_W])
	w = np.tile(np.array(range(cfg.IMAGE.OUTPUT_W))[np.newaxis, :], [cfg.IMAGE.OUTPUT_H, 1])
	hw_grid = np.stack((h, w), axis=-1)
	hw_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.ANCHORS_NUM, 2]
	hw_grid = np.tile(hw_grid, cfg.IMAGE.ANCHORS_NUM).reshape(hw_shape) 
	box_shape = [cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.ANCHORS_NUM, cfg.IMAGE.BBOX_DIM+1+1]
	imglabel = imglabel[..., 1:].reshape(box_shape)
	imglabel = np.concatenate((imglabel, hw_grid, anchors), axis=-1)
	preds = imglabel[imglabel[..., 0]>0.1]
	objness = preds[..., 0]
	cls_idx = preds[..., 1]
	box = preds[..., 2:]
	x = (box[:, 0] + box[:, -4]) * cfg.IMAGE.STRIDE / cfg.IMAGE.H_SCALE_RATIO 
	y = (box[:, 1] + box[:, -3]) * cfg.IMAGE.STRIDE / cfg.IMAGE.W_SCALE_RATIO
	h = box[:, 2] / cfg.IMAGE.H_SCALE_RATIO * box[:, -2]
	w = box[:, 3] / cfg.IMAGE.W_SCALE_RATIO * box[:, -1]
	left = y - w / 2
	top = x - h / 2
	right = y + w / 2
	bottom = x + h / 2
	return np.stack([cls_idx, objness, left, top, right, bottom], axis=-1)




lidar_preprocessor  = preprocess.LidarPreprocessor()
trainset            = dataset.Dataset(lidar_preprocessor, 'train')
pbar                = tqdm(trainset)
img_anchors         = loader.load_anchors(cfg.IMAGE.ANCHORS)
bev_anchors         = loader.load_anchors(cfg.BEV.ANCHORS)
img_dir             = os.path.join(cfg.CONTFUSE.DATASETS_DIR, "image_2")
lidar_dir           = os.path.join(cfg.CONTFUSE.DATASETS_DIR, "lidar_files")
for data in pbar:
	vis_tools.imshow_image(data[0][0][..., -1])
	vis_tools.imshow_image(data[1][0])
	vis_tools.imshow_image(data[6][0][..., 0])
	vis_tools.imshow_image(data[7][0][..., 0])
	bevlabel = parse_bevlabel(data[6][0], bev_anchors)
	imglabel = parse_imglabel(data[7][0], img_anchors)
	img_bboxes = postprocess.img_nms(imglabel, cfg.IMAGE.IOU_THRESHOLD)
	bev_bboxes = postprocess.bev_nms(bevlabel, cfg.BEV.DISTANCE_THRESHOLDS)
	img_file = os.path.join(img_dir, data[9][0]+'.png')
	img = cv2.imread(img_file)
	vis_tools.imshow_img_bbox(img, np.array(img_bboxes))
	vis_tools.imshow_bev_bbox(data[0][0][..., -3:], np.array(bev_bboxes))
	


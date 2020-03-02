import sys
sys.path.append("../")
import os
import cv2
from glob import glob
from utils import math
from utils import vis_tools
from config.config import cfg
from data import postprocess
from data import loader
import numpy as np


img_pred_files      = glob(cfg.CONTFUSE.LOG_DIR+"/pred/img_pred/*")
bev_pred_files      = glob(cfg.CONTFUSE.LOG_DIR+"/pred/bev_pred/*")
img_anchors         = loader.load_anchors(cfg.IMAGE.ANCHORS)
bev_anchors         = loader.load_anchors(cfg.BEV.ANCHORS)
img_dir             = os.path.join(cfg.CONTFUSE.DATASETS_DIR, "image_2/")
lidar_dir           = os.path.join(cfg.CONTFUSE.DATASETS_DIR, "lidar_files/")


for fi in bev_pred_files:
  bev = np.zeros([640, 640, 3], dtype=np.float32)
  bev_pred = np.load(fi)
  vis_tools.imshow_image(math.sigmoid(bev_pred[..., 0]))
  bev_pred_cls = math.sigmoid(bev_pred[..., 1:cfg.CONTFUSE.CLASSES_NUM+1])
  # vis_tools.imshow_image(math.sigmoid(bev_pre 
  bev_bboxes = postprocess.parse_bev_predmap(bev_pred, bev_anchors)
  bev_bboxes = postprocess.bev_nms(bev_bboxes, cfg.BEV.DISTANCE_THRESHOLDS)
  vis_tools.imshow_bev_bbox(bev, np.array(bev_bboxes))
  # vis_tools.imshow_image(bev_pred[..., 0])


for fi in img_pred_files:
  img_pred = np.load(fi)
  img_map = img_pred.reshape([cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, 6, 11])
  vis_tools.imshow_image(np.max(math.sigmoid(img_map[..., 0]), axis=-1))
  vis_tools.imshow_image(np.max(math.sigmoid(img_map[..., 1:cfg.CONTFUSE.CLASSES_NUM+1])[..., 0], axis=-1))
  img_bboxes = postprocess.parse_img_predmap(img_pred, img_anchors)
  img_bboxes = postprocess.img_nms(img_bboxes, cfg.IMAGE.IOU_THRESHOLD)
  img_file = img_dir + fi[-14:-8] + ".png"  
  img = cv2.imread(img_file)
  vis_tools.imshow_img_bbox(img, np.array(img_bboxes))
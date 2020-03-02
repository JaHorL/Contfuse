import cv2
import numpy as np
from config.config import cfg
from utils import transform


def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split())!= 0]
    np.random.shuffle(annotations)
    return annotations
  

def load_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readlines()
    new_anchors = np.zeros([len(anchors), len(anchors[0].split())], dtype=np.float32)
    for i in range(len(anchors)):
        new_anchors[i] = np.array(anchors[i].split(), dtype=np.float32)
    return new_anchors


def load_calib(calib_file):

    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)
    obj = lines[0].strip().split(' ')[1:]
    P00 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P10 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P20 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P30 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    tr_lidar2cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    tr_imu2lidar = np.array(obj, dtype=np.float32)
    return  P20.reshape(3, 4),R0.reshape(3, 3),tr_lidar2cam.reshape(3, 4)


def load_lidar(lidar_file):
	lidar = np.fromfile(lidar_file, np.float32).reshape((-1, 4))
	return lidar


def load_label(label_file):
	with open(label_file, "r") as f:
		lines = f.read().split("\n")
	types = []
	dimensions = []
	box2d_corners = []
	locations = []
	rzs = []

	for line in lines:
		if not line:
			continue
		line = line.split(" ")
		if(line[0] not in cfg.CONTFUSE.CLASSES_LIST):
			continue
		types.append(cfg.CONTFUSE.CLASSES_LIST.index(line[0]))
		dimensions.append(np.array(line[8:11]).astype(np.float32))
		box2d_corners.append(np.array(line[4:8]).astype(np.float32))
		locations.append(np.array(line[11:14]).astype(np.float32))
		rzs.append(float(line[14]))
	return types, dimensions, box2d_corners, locations, rzs


def load_image(image_file):
	img = cv2.imread(image_file)
	out_img = cv2.resize(img, (cfg.IMAGE.INPUT_W, cfg.IMAGE.INPUT_H), cv2.INTER_CUBIC)
	return out_img

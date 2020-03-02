import numpy as np
from config.config import cfg


def bevbox_compose(x, y, w, l, rz):
    center_mat = np.array([[y, y, y, y], 
                           [x, x, x, x]])
    tracklet_box = np.array([[w / 2, -w / 2, -w / 2, w / 2],
                             [-l / 2, -l / 2, l / 2, l / 2]])
    yaw = -rz  
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
    corner_pos_in_lidar = np.dot(rot_mat, tracklet_box)+center_mat
    bevbox = corner_pos_in_lidar.transpose().astype(np.int32)
    return bevbox


def box3d_compose(location, dimension, rz):
	box3d = np.zeros((8,3), dtype=np.float32)
	x, y, z = location
	h, w, l = dimension
	center_mat = np.array([[y, y, y, y], 
                           [x, x, x, x]])
	tracklet_box = np.array([[w / 2, -w / 2, -w / 2, w / 2],
							[-l / 2, -l / 2, l / 2, l / 2]])
	yaw = -rz  
	rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
						[np.sin(yaw), np.cos(yaw)]])
	corner_pos_in_lidar = np.dot(rot_mat, tracklet_box)+center_mat
	box = corner_pos_in_lidar.transpose().astype(np.int32)
	bottom = z-h/2
	top = z+h/2
	box3d[:4, :2] = box
	box3d[:4, 2] = bottom
	box3d[4:, :2] = box
	box3d[:4, 2] = top
	return box3d

    
def location_lidar2bev(location):
    location[0] = (cfg.BEV.X_MAX - location[0]) / cfg.BEV.X_RESOLUTION
    location[1] = (cfg.BEV.Y_MAX - location[1]) / cfg.BEV.Y_RESOLUTION
    location[2] = (cfg.BEV.Z_MAX - location[2]) / cfg.BEV.Z_RESOLUTION
    return location


def location_lidar2bevlabel(location):
    location[0] = (cfg.BEV.X_MAX - location[0]) / (cfg.BEV.X_RESOLUTION  * cfg.BEV.STRIDE)
    location[1] = (cfg.BEV.Y_MAX - location[1]) / (cfg.BEV.Y_RESOLUTION  * cfg.BEV.STRIDE)
    location[2] = (cfg.BEV.Z_MAX - location[2]) / cfg.BEV.Z_RESOLUTION
    return location


def bbox3d_lidar2img(lidar_bboxes3d, pp, r0):
	num = len(lidar_bboxes3d)
	img_bboxes3d = np.zeros((num, 8, 2)).astype(np.int8)
	for i in range(num):
		lidar_bbox = lidar_bboxes3d[i]
		tmp = np.ones([4])
		tmp[:3] = lidar_bbox
		img_bbox = (pp.dot(r0)).dot(tmp)
		img_bboxes3d[i] = np.array([img_bbox[0]/img_bbox[2], img_bbox[1]/img_bbox[2]]).astype(np.int8)
	return img_bboxes3d


def location_cam2lidar(location, tr):
	location_in_cam = np.ones([4], dtype=np.float32)
	location_in_cam[:3] = location 
	t = np.zeros([4, 4], dtype=np.float32)
	t[:3, :] = tr
	t[3, 3] = 1
	t_inv = np.linalg.inv(t)
	location_in_lidar = t_inv.dot(location_in_cam)
	location_in_lidar = location_in_lidar[:3]
	return location_in_lidar


def location_lidar2cam(location, tr):
	location_in_lidar = np.ones([4], dtype=np.float32)
	location_in_lidar[:3] = location
	t = np.zeros([4, 4], dtype=np.float32)
	t[:3, :] = tr
	t[3, 3] = 1
	location_in_cam = location_in_lidar.dot(t)
	location_in_cam = location_in_cam[:3]
	return location_in_cam


def get_tr_lidar2img(p20, r0, tr_lidar2cam):
	t_p20 = np.zeros([4,4], dtype=float)
	t_r0 = np.zeros([4,4], dtype=float)
	t_tr_lidar2cam = np.zeros([4,4], dtype=float)
	t_p20[:3,:] = p20
	t_r0[:3,:3] = r0
	t_tr_lidar2cam[:3,:] = tr_lidar2cam
	t_p20[3][3] = 1
	t_r0[3][3] = 1
	t_tr_lidar2cam[3][3] = 1
	tr = (t_p20.dot(t_r0)).dot(t_tr_lidar2cam)
	return tr 


def ry_to_rz(ry):
	ry = np.array(ry).astype(np.float32)
	angle = -ry - np.pi / 2
	return angle


def rz_to_ry(rz):
	rz = np.array(rz).astype(np.float32)
	angle = -rz - np.pi / 2
	return angle

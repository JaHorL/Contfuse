import cv2
import numpy as np
from utils import transform
from config.config import cfg



def create_img_label(types, box2d_corners, anchors):
    def iou_wh(r1, r2):
        min_w = min(r1[0],r2[0])
        min_h = min(r1[1],r2[1])
        area_r1 = r1[0]*r1[1]
        area_r2 = r2[0]*r2[1]	
        intersect = min_w * min_h		
        union = area_r1 + area_r2 - intersect
        return intersect/union

    def get_active_anchors(roi, anchors):
        indxs = []
        iou_max, index_max = 0, 0
        for i,a in enumerate(anchors):
            iou = iou_wh(roi, a)
            if iou>0.5:
                indxs.append(i)
            if iou > iou_max:
                iou_max, index_max = iou, i
        if len(indxs) == 0:
            indxs.append(index_max)
        return indxs

    obj_num = len(types)
    s = 1 + 1 + cfg.IMAGE.BBOX_DIM    
    label = np.zeros((cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.LABEL_Z), dtype=np.float32)
    for i in range(obj_num):
        h = (box2d_corners[i][3]-box2d_corners[i][1]) * cfg.IMAGE.H_SCALE_RATIO
        w = (box2d_corners[i][2]-box2d_corners[i][0]) * cfg.IMAGE.W_SCALE_RATIO 
        center_h = (box2d_corners[i][3]+box2d_corners[i][1])/2 * cfg.IMAGE.H_SCALE_RATIO
        center_w = (box2d_corners[i][2]+box2d_corners[i][0])/2 * cfg.IMAGE.W_SCALE_RATIO
        grid_h = int(center_h / cfg.IMAGE.STRIDE)
        grid_w = int(center_w / cfg.IMAGE.STRIDE)
        grid_h_offset = center_h / cfg.IMAGE.STRIDE - grid_h
        grid_w_offset = center_w / cfg.IMAGE.STRIDE - grid_w
        active_idxs = get_active_anchors([h, w], anchors)
        label[grid_h, grid_w, 0] = 1
        for idx in active_idxs:
            dh = h / anchors[idx][0]
            dw = w / anchors[idx][1]
            label[grid_h, grid_w, s*idx+1:s*(idx+1)+1] = np.array([1, types[i], grid_h_offset, grid_w_offset, dh, dw])
    return label


def create_bev_label(locations, dimensions, rys, types, tr, anchors):
    obj_num = len(types)
    objectness_class_map = np.zeros((cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, 3), dtype=np.float32)
    bev_center_map = np.zeros((cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, 3), dtype=np.float32)
    rzs = transform.ry_to_rz(rys)
    for i in range(obj_num):
        location = transform.location_cam2lidar(locations[i], tr)
        bev_location = transform.location_lidar2bevlabel(location)
        xx, yy, _ = bev_location
        hwl = np.array(dimensions[i]) / (cfg.BEV.X_RESOLUTION * 4)
        box = transform.bevbox_compose(xx, yy, hwl[1], hwl[2], rzs[i])
        cv2.fillConvexPoly(objectness_class_map, box, [i+1, types[i], 0.0])
        cv2.fillConvexPoly(bev_center_map, box, [float(xx), float(yy), 0.0])
    bev_label = np.zeros([cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z], np.float32)
    for i in range(cfg.BEV.OUTPUT_X):
        for j in range(cfg.BEV.OUTPUT_Y):
            if objectness_class_map[i][j][0] < 0.1: continue
            type_id = int(objectness_class_map[i][j][1])
            idx = int(objectness_class_map[i][j][0]-1)
            rz = rzs[idx]
            dim = dimensions[idx]
            theta = rz if rz < 0 else rz + 3.14
            center_x, center_y, _ = bev_center_map[i][j]
            delta_x = (i-center_x) / anchors[type_id][3]
            delta_y = (j-center_y) / anchors[type_id][4]
            offset_xy = np.sqrt(pow((center_x-i), 2) + pow((center_y-j), 2))
            prob = pow(cfg.BEV.PROB_DECAY, offset_xy)
            h, w, l= dim / anchors[type_id][:3]
            box = np.array([delta_x, delta_y, h, w, l, theta], np.float32)
            bev_label[i][j][:3] = np.array([1, type_id, prob])
            bev_label[i][j][3+type_id*cfg.BEV.BBOX_DIM:3+(type_id+1)*cfg.BEV.BBOX_DIM] = box
    return bev_label





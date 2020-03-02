
import sys
sys.path.append("..")
import os
import numpy as np
from sklearn.cluster import KMeans
from config.config import cfg



class statistic(object):
    def __init__(self):
        self.anno_path         = cfg.CONTFUSE.TRAIN_DATA
        self.class_list        = cfg.CONTFUSE.CLASSES_LIST
        self.annotations       = self.load_annotations(self.anno_path)


    def load_annotations(self, annot_path):
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split())!= 0]
        np.random.shuffle(annotations)
        return annotations

    def statist_all_labels(self):
        total_types = []
        total_dimensions = []
        total_box2d_corners = []
        total_box3d_locations = []
        total_rzs = []
        for fi in self.annotations:
            label_file = fi.split()[2]
            types, dimensions, box2d_corners, box3d_locations, rzs = self.load_label(label_file)
            total_types += types
            total_dimensions += dimensions
            total_box2d_corners += box2d_corners
            total_box3d_locations += box3d_locations
            total_rzs += rzs
        img_anchors = self.analysis_box2d(total_box2d_corners)
        bev_anchors = self.analysis_box3d(total_dimensions, total_types)
        loss_scale = self.analysis_types(total_types, bev_anchors)


    def load_label(self, label_file):
        with open(label_file, "r") as f:
            lines = f.read().split("\n")
        types = []
        dimensions = []
        box2d_corners = []
        box3d_locations = []
        rzs = []
        for line in lines:
            if not line:
                continue
            line = line.split(" ")
            if(line[0] not in self.class_list):
                continue
            types.append(self.class_list.index(line[0]))
            dimensions.append(np.array(line[8:11]).astype(np.float32))
            box2d_corners.append(np.array(line[4:8]).astype(np.float32))
            box3d_locations.append(np.array(line[11:14]).astype(np.float32))
            rzs.append(float(line[14]))
        return types, dimensions, box2d_corners, box3d_locations, rzs


    def analysis_box2d(self, box2d_corners):
        box_hw = np.zeros([len(box2d_corners), 2], dtype=np.float32)
        count = 0
        for corner in box2d_corners:
            h = corner[3]-corner[1]
            w = corner[2]-corner[0]
            box_hw[count] = [h, w]
            count += 1
        cluster = KMeans(n_clusters=6)
        cluster.fit(box_hw)
        print("cluster_center: ", cluster.cluster_centers_)
        return cluster.cluster_centers_

    def calc_mean_dxdy(self, w, l):
        ww = int(w / (cfg.BEV.X_RESOLUTION * 2))
        ll = int(l / (cfg.BEV.X_RESOLUTION * 2))
        dx_sum = 0
        dy_sum = 0
        for i in range(ww):
            dx_sum += i
        dx_mean = dx_sum / ww
        for i in range(ll):
            dy_sum += i
        dy_mean = dy_sum / ll
        return np.array([dx_mean, dy_mean], dtype=np.float32)

    def analysis_box3d(self, box3d_dimensions, types):
        total_box_hwl = np.zeros([len(self.class_list), 4])
        mean_box_hwldxdy = np.zeros([len(self.class_list), 5])
        for i in range(len(box3d_dimensions)):
            total_box_hwl[types[i]][:3] += box3d_dimensions[i]
            total_box_hwl[types[i]][3] += 1
        for i in range(len(total_box_hwl)):
            h, w, l = total_box_hwl[i][:3] / total_box_hwl[i][3]
            dx, dy = self.calc_mean_dxdy(w, l)
            mean_box_hwldxdy[i] = np.array([h, w, l, dx, dy], dtype=np.float32)
        print("mean_box_hwldxdy: ", mean_box_hwldxdy)
        return mean_box_hwldxdy

    def analysis_types(self, types, bev_anchors):
        types_set = set(types)
        types = np.array(types)
        types_num_array = np.zeros(len(types_set))
        for i in types_set:
            types_num_array[i] = len(types[types==i])
        img_ratio = 1 / (types_num_array / np.sum(types_num_array))     
        img_scale = img_ratio / img_ratio[0]
        bev_types_num = types_num_array * bev_anchors[:, 1] * bev_anchors[:, 2]
        bev_ratio = 1 / ( bev_types_num / np.sum(bev_types_num))
        bev_scale = bev_ratio / bev_ratio[0]
        print(bev_ratio, bev_scale)
        return img_scale, bev_scale

if __name__ == "__main__":
    sat = statistic()
    sat.statist_all_labels()
    # print(sat.annotations[0])
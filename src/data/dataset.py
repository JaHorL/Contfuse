import os
import cv2
import time
import ctypes
import threading
import numpy as np
from config.config import cfg
from utils import utils
from utils import vis_tools
from utils import timer
from utils import transform
from data import loader
from data import labels
from data import preprocess


class Dataset(object):
    def __init__(self, preprocessor, dataset_type):
        if dataset_type == 'train':
            self.anno_path     = cfg.CONTFUSE.TRAIN_DATA
            self.batch_size    = cfg.TRAIN.BATCH_SIZE
            self.is_data_aug   = cfg.TRAIN.IS_DATA_AUG
        if dataset_type == 'val':
            self.anno_path     = cfg.CONTFUSE.VAL_DATA 
            self.batch_size    = cfg.EVAL.BATCH_SIZE
            self.is_data_aug   = False
        if dataset_type == 'test':
            self.anno_path     = cfg.CONTFUSE.TEST_DATA
            self.batch_size    = cfg.EVAL.BATCH_SIZE
            self.is_data_aug   = False


        self.img_anchors       = loader.load_anchors(cfg.IMAGE.ANCHORS)
        self.bev_anchors       = loader.load_anchors(cfg.BEV.ANCHORS)
        self.annotations       = loader.load_annotations(self.anno_path)
        self.num_samples       = len(self.annotations)
        self.num_batchs        = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count       = 0
        self.is_use_thread     = cfg.CONTFUSE.IS_USE_THREAD

        self.cuda_preprocessor = preprocessor.preprocessor
  

        self.loader_need_exit = 0
        self.timer = timer.Timer()
        
        if self.is_use_thread:
            self.prepr_data = []
            self.max_cache_size = 10
            self.lodaer_processing =  threading.Thread(target=self.loader)
            self.lodaer_processing.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit = True
        print('set loader_need_exit True')
        self.lodaer_processing.join()
        print('exit lodaer_processing')

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_count < self.num_batchs:
            self.batch_count += 1
            return self.load() 
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration



    def preprocess_data(self):
        batch_bev = np.zeros((self.batch_size, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, cfg.BEV.INPUT_Z), dtype=np.float32)
        batch_img = np.zeros((self.batch_size, cfg.IMAGE.INPUT_H, cfg.IMAGE.INPUT_W, 3), dtype=np.float32)
        batch_mapping1x = np.zeros((self.batch_size, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, 2), dtype=np.int32)
        batch_mapping2x = np.zeros((self.batch_size, int(cfg.BEV.INPUT_X/2), int(cfg.BEV.INPUT_Y/2), 2), dtype=np.int32)
        batch_mapping4x = np.zeros((self.batch_size, int(cfg.BEV.INPUT_X/4), int(cfg.BEV.INPUT_Y/4), 2), dtype=np.int32)
        batch_mapping8x = np.zeros((self.batch_size, int(cfg.BEV.INPUT_X/8), int(cfg.BEV.INPUT_Y/8), 2), dtype=np.int32)
        batch_bev_label = np.zeros((self.batch_size, cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z), dtype=np.float32)
        batch_img_label = np.zeros((self.batch_size, cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.LABEL_Z), dtype=np.float32)
        batch_tr = np.zeros((self.batch_size, 3, 4), dtype=np.float32)

        batch_frame_id = []
        num = 0 
        while num < self.batch_size:
            index = self.batch_count * self.batch_size + num
            if index == self.num_samples: index=0
            annotation = self.annotations[index]
            if not annotation: continue
            lidar_file , image_file, label_file, calib_file  = annotation.split()
            frame_id = lidar_file[-10:-4]
            p20, r0, tr_lidar2cam = loader.load_calib(calib_file)
            img = loader.load_image(image_file)
            point_cloud = loader.load_lidar(lidar_file)
            bev, mapping1x, mapping2x, mapping4x, mapping8x = preprocess.lidar_preprocess(point_cloud,
                                                              p20, r0, tr_lidar2cam, self.cuda_preprocessor)
            types, dimensions, box2d_corners, locations, rzs = loader.load_label(label_file)
            bev_label = labels.create_bev_label(locations, dimensions, rzs, types, tr_lidar2cam, self.bev_anchors)
            img_label = labels.create_img_label(types, box2d_corners, self.img_anchors)
            # vis_tools.imshow_image(img)
            batch_bev[num, ...] = bev
            batch_img[num, ...] = img / 255.0
            batch_mapping1x[num, ...] = mapping1x
            batch_mapping2x[num, ...] = mapping2x
            batch_mapping4x[num, ...] = mapping4x
            batch_mapping8x[num, ...] = mapping8x
            batch_bev_label[num, ...] = bev_label
            batch_img_label[num, ...] = img_label
            batch_tr[num, ...] = tr_lidar2cam
            batch_frame_id.append(frame_id)
            num += 1
        return (batch_bev, batch_img, batch_mapping1x, batch_mapping2x, batch_mapping4x, 
                batch_mapping8x, batch_bev_label, batch_img_label, batch_tr, batch_frame_id)


    def loader(self):
        while(not self.loader_need_exit):
            if len(self.prepr_data) < self.max_cache_size: 
                self.prepr_data = self.preprocess_data() + self.prepr_data
            else:
                time.sleep(0.1)
                self.loader_need_exit = False

    def load(self):
        if self.is_use_thread:
            while len(self.prepr_data) == 0:
                time.sleep(0.1)
            data_ori = self.prepr_data.pop()
        else:
            data_ori = self.preprocess_data()
        return data_ori

                                                          

if __name__ == "__main__":
    pass
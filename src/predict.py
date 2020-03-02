import os
import io
import time
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from config.config import cfg
from utils import utils
from data import dataset
from data import preprocess
from data import postprocess
from data import loader
from utils import vis_tools
from models import contfuse_network





class predicter(object):

    def __init__(self):
        self.initial_weight      = cfg.EVAL.WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.CONTFUSE.MOVING_AVE_DECAY
        self.eval_logdir        = "./data/logs/eval"
        self.lidar_preprocessor  = preprocess.LidarPreprocessor()
        self.evalset             = dataset.Dataset(self.lidar_preprocessor, 'test')
        self.output_dir          = cfg.EVAL.OUTPUT_PRED_PATH
        self.img_anchors         = loader.load_anchors(cfg.IMAGE.ANCHORS )
        self.bev_anchors         = loader.load_anchors(cfg.BEV.ANCHORS)

        with tf.name_scope('model'):
            self.model               = contfuse_network.ContfuseNetwork()
            self.net                 = self.model.load()
            self.img_pred            = self.net['img_pred']
            self.bev_pred            = self.net['bev_pred']

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver()#ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.initial_weight)


    def predict(self):
        bev_imwrite_path = os.path.join(self.output_dir, "bev_imshow_result/")
        img_imwrite_path = os.path.join(self.output_dir, "img_imshow_result/")
        bev_result_path  = os.path.join(self.output_dir, "bev_result/")
        img_result_path  = os.path.join(self.output_dir, "img_result/")
        img_dir          = os.path.join(cfg.CONTFUSE.DATASETS_DIR, "image_2/")
        if os.path.exists(bev_imwrite_path):
            shutil.rmtree(bev_imwrite_path)
        os.mkdir(bev_imwrite_path)
        if os.path.exists(img_imwrite_path):
            shutil.rmtree(img_imwrite_path)
        os.mkdir(img_imwrite_path)
        if os.path.exists(bev_result_path):
            shutil.rmtree(bev_result_path)
        os.mkdir(bev_result_path)
        if os.path.exists(img_result_path):
            shutil.rmtree(bev_result_path)
        os.mkdir(bev_result_path)
        for epoch in range(len(self.evalset)):
            eval_data = next(self.evalset)
            frame_id = eval_data[9][0]
            tr = eval_data[8]
            img_pred, bev_pred = self.sess.run([self.img_pred, self.bev_pred],
                                              feed_dict={self.net["bev_input"]:   eval_data[0],
                                                         self.net["img_input"]:   eval_data[1],
                                                         self.net["mapping1x"]:   eval_data[2],
                                                         self.net["mapping2x"]:   eval_data[3],
                                                         self.net["mapping4x"]:   eval_data[4],
                                                         self.net["mapping8x"]:   eval_data[5],
                                                         self.net["trainable"]:   True
                                                         })
            bev_bboxes = postprocess.parse_bev_predmap(bev_pred[0], self.bev_anchors)
            bev_bboxes = postprocess.bev_nms(bev_bboxes, cfg.BEV.DISTANCE_THRESHOLDS)
            img_bboxes = postprocess.parse_img_predmap(img_pred[0], self.img_anchors)
            img_bboxes = postprocess.img_nms(img_bboxes, cfg.IMAGE.IOU_THRESHOLD)
            postprocess.save_lidar_results(bev_bboxes, tr, frame_id[0], bev_result_path)
            postprocess.save_image_results(img_bboxes, frame_id, img_result_path)
            vis_tools.imwrite_bev_bbox(eval_data[0][0][..., -3:]*200, bev_bboxes, bev_imwrite_path, frame_id)
            img_file = os.path.join(img_dir, frame_id+".png")  
            img = cv2.imread(img_file)
            vis_tools.imwrite_img_bbox(img, img_bboxes, img_imwrite_path, frame_id)
            print("{}/{}, bev bboxes:\n".format(epoch, len(self.evalset)), bev_bboxes)

if __name__ == "__main__":
    predicter = predicter()
    predicter.predict()
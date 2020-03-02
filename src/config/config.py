import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C                           = edict()
cfg                           = __C


__C.CONTFUSE                  = edict()
__C.CONTFUSE.CLASSES_LIST     = ['Car','Van','Truck','Pedestrian','Cyclist','Misc']
__C.CONTFUSE.CLASSES_COLOR    = [(255,0,0),(255,255,0),(255,0,255),(0,255,0),(128,64,255),(0,255,255)]
__C.CONTFUSE.CLASSES_NUM      = len(__C.CONTFUSE.CLASSES_LIST)
__C.CONTFUSE.EPSILON          = 0.00001
__C.CONTFUSE.MAX_PTS_NUM      = 200000
__C.CONTFUSE.ROOT_DIR         = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
__C.CONTFUSE.LOG_DIR          = osp.join(__C.CONTFUSE.ROOT_DIR, 'logs')
__C.CONTFUSE.DATASETS_DIR     = "/home/ljh/dataset/detection_3d/kitti_compitetion"
__C.CONTFUSE.TRAIN_DATA       = osp.join(__C.CONTFUSE.DATASETS_DIR, "training.txt")
__C.CONTFUSE.VAL_DATA         = osp.join(__C.CONTFUSE.DATASETS_DIR, "val.txt")
__C.CONTFUSE.TEST_DATA        = osp.join(__C.CONTFUSE.DATASETS_DIR, "testing.txt")
__C.CONTFUSE.PREPROCESS_LIB   = osp.join(__C.CONTFUSE.ROOT_DIR, 'src/utils/liblidar_preprocessor.a')
__C.CONTFUSE.MOVING_AVE_DECAY = 0.9995
__C.CONTFUSE.IS_USE_THREAD    = False


__C.BEV                       = edict()
__C.BEV.ANCHORS               = __C.CONTFUSE.ROOT_DIR + "/src/config/anchors/bev_anchors.txt"
# __C.BEV.LOSS_SCALE            = np.array([1.00, 6.47, 6.37, 72.97, 107.18, 35.13])
__C.BEV.LOSS_SCALE            = np.array([1.00, 5.00, 5.00, 20.0, 20.0, 10.0])
__C.BEV.X_MAX                 = 80
__C.BEV.X_MIN                 = 0
__C.BEV.Y_MAX                 = 40
__C.BEV.Y_MIN                 = -40
__C.BEV.Z_MAX                 = 1
__C.BEV.Z_MIN                 = -2.5     
__C.BEV.X_RESOLUTION          = 0.125
__C.BEV.Y_RESOLUTION          = 0.125
__C.BEV.Z_RESOLUTION          = 0.5
__C.BEV.Z_STATISTIC_DIM       = 6
__C.BEV.STRIDE                = 4
__C.BEV.BBOX_DIM              = 6
__C.BEV.PROB_DECAY            = 0.98
__C.BEV.IS_LIDAR_AUG          = False
__C.BEV.INPUT_X               = int((__C.BEV.X_MAX - __C.BEV.X_MIN) / __C.BEV.X_RESOLUTION)
__C.BEV.INPUT_Y               = int((__C.BEV.Y_MAX - __C.BEV.Y_MIN) / __C.BEV.Y_RESOLUTION)
__C.BEV.LAYERED_DIM           = int((__C.BEV.Z_MAX - __C.BEV.Z_MIN)/ __C.BEV.Z_RESOLUTION)
__C.BEV.INPUT_Z               = __C.BEV.LAYERED_DIM + __C.BEV.Z_STATISTIC_DIM
__C.BEV.LABEL_Z               = int(1 + 1 + 1 + __C.BEV.BBOX_DIM * __C.CONTFUSE.CLASSES_NUM)
__C.BEV.OUTPUT_X              = int(__C.BEV.INPUT_X / __C.BEV.STRIDE)
__C.BEV.OUTPUT_Y              = int(__C.BEV.INPUT_Y / __C.BEV.STRIDE)
__C.BEV.DISTANCE_THRESHOLDS   = [1.5, 3.0, 3.0, 1.0, 1.0, 1.5]


__C.IMAGE                     = edict()
__C.IMAGE.ANCHORS             = __C.CONTFUSE.ROOT_DIR + "/src/config/anchors/image_anchors.txt"
# __C.IMAGE.LOSS_SCALE          = np.array([1.00, 9.87, 26.22, 6.41, 17.81, 29.92])
__C.IMAGE.LOSS_SCALE          = np.array([1.00, 5.00, 10.0, 5.00, 10.0, 10.0])
__C.IMAGE.INPUT_H             = 192
__C.IMAGE.INPUT_W             = 640
__C.IMAGE.H_SCALE_RATIO       = __C.IMAGE.INPUT_H / 375
__C.IMAGE.W_SCALE_RATIO       = __C.IMAGE.INPUT_W / 1242
__C.IMAGE.BBOX_DIM            = 4
__C.IMAGE.STRIDE              = 8
__C.IMAGE.IS_IMG_AUG          = False
__C.IMAGE.OUTPUT_H            = int(__C.IMAGE.INPUT_H/ __C.IMAGE.STRIDE)
__C.IMAGE.OUTPUT_W            = int(__C.IMAGE.INPUT_W / __C.IMAGE.STRIDE)
__C.IMAGE.ANCHORS_NUM         = 6
__C.IMAGE.LABEL_Z             = int(1 + (__C.IMAGE.BBOX_DIM + 1 + 1) * __C.IMAGE.ANCHORS_NUM)
__C.IMAGE.IOU_THRESHOLD       = 0.5

__C.TRAIN                     = edict()

__C.TRAIN.PRETRAIN_WEIGHT     = "../checkpoint/contfuse_val_loss=1266.4186.ckpt-10"
__C.TRAIN.SAVING_STEPS        = 6000
__C.TRAIN.BATCH_SIZE          = 1
__C.TRAIN.FRIST_STAGE_EPOCHS  = 1
__C.TRAIN.SECOND_STAGE_EPOCHS = 15
__C.TRAIN.WARMUP_EPOCHS       = 0
__C.TRAIN.LEARN_RATE_INIT     = 1e-3
__C.TRAIN.LEARN_RATE_END      = 1e-5
__C.TRAIN.IS_DATA_AUG         = True


__C.EVAL                      = edict()
__C.EVAL.BATCH_SIZE           = 1
__C.EVAL.WEIGHT               = "../checkpoint/contfuse_val_loss=1266.4186.ckpt-10"
__C.EVAL.OUTPUT_GT_PATH       = osp.join(__C.CONTFUSE.LOG_DIR, "gt")
__C.EVAL.OUTPUT_PRED_PATH     = osp.join(__C.CONTFUSE.LOG_DIR, "pred")



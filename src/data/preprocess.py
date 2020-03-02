import numpy as np
from config.config import cfg
from data.lidar_preprocess.build import libcuda_preprocessor
from utils import transform

class LidarPreprocessor():
    def __init__(self):
        self.preprocessor = libcuda_preprocessor.Preprocessor()
        self.preprocessor.PreprocessorInit(cfg.BEV.X_MIN, cfg.BEV.X_MAX,
                                           cfg.BEV.Y_MIN, cfg.BEV.Y_MAX,
                                           cfg.BEV.Z_MIN, cfg.BEV.Z_MAX,
                                           cfg.BEV.X_RESOLUTION, cfg.BEV.Y_RESOLUTION,
                                           cfg.BEV.Z_RESOLUTION, cfg.BEV.Z_STATISTIC_DIM,
                                           cfg.IMAGE.INPUT_H, cfg.IMAGE.INPUT_W,
                                           cfg.IMAGE.H_SCALE_RATIO, cfg.IMAGE.W_SCALE_RATIO)




def lidar_preprocess(point_cloud, p20, r0, tr_lidar2cam, cuda_preprocessor):
	tr_lidar2img = transform.get_tr_lidar2img(p20, r0, tr_lidar2cam).astype(np.float32)
	cuda_preprocessor.PreprocessData(point_cloud, tr_lidar2img, int(point_cloud.shape[0]))
	bev = cuda_preprocessor.GetBev().astype(np.float32)
	mapping1x = cuda_preprocessor.GetMapping1x()
	mapping2x = cuda_preprocessor.GetMapping2x()
	mapping4x = cuda_preprocessor.GetMapping4x()
	mapping8x = cuda_preprocessor.GetMapping8x()
	return bev, mapping1x, mapping2x, mapping4x, mapping8x

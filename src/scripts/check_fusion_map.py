import sys
sys.path.append("../")
import numpy as np
import cv2
from data import dataset
from data import preprocess
from utils import vis_tools
from tqdm import tqdm


def project_fusionmap_to_img(img, fusionmap, down_ratio):
	points = fusionmap[fusionmap[..., 0] > 0]
	new_size = (int(img.shape[1]/down_ratio),int(img.shape[0]/down_ratio))
	img = cv2.resize(img, new_size)
	for p in points:
		img[p[1]][p[0]] = 1.0
	vis_tools.imshow_image(img)
	vis_tools.imshow_image(fusionmap[..., 0].astype(np.float32))


if __name__ == "__main__":
	lidar_preprocessor  = preprocess.LidarPreprocessor()
	trainset            = dataset.Dataset(lidar_preprocessor, 'train')
	pbar                = tqdm(trainset)
	for data in pbar:
		img = data[1][0]
		# print(img)
		mapping1x = data[2][0]
		mapping2x = data[3][0]
		mapping4x = data[4][0]
		mapping8x = data[5][0]
		project_fusionmap_to_img(img, mapping1x, 1)
		# vis_tools.imshow_image(img)
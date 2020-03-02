import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def softmax(x):
	e_x = np.exp(x)
	return e_x/np.sum(e_x, axis=-1)[..., np.newaxis]
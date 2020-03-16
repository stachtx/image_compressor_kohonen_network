import numpy as np


class Compressor():

    def mse(image_a, image_b):
        # calculate mean square error between two images
        err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
        err /= float(image_a.shape[0] * image_a.shape[1])

        return err

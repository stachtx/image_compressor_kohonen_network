import cv2
import imageio
import numpy as np

from Config import Config


class ImageHandler:

    def __init__(self):
        self.config = Config().imageHandlerConfig
        self.image = cv2.imread(self.config.image_location, cv2.IMREAD_GRAYSCALE)
        self.image_height = len(self.image)
        self.image_width = len(self.image[0])
        self.vector_dimension = self.config.block_width * self.config.block_height

    def split_image_to_blocks(self):
        image_vectors = []
        for i in range(0, self.image_height, self.config.block_height):
            for j in range(0, self.image_width, self.config.block_width):
                image_vectors.append(
                    np.reshape(
                        self.image[i:i + self.config.block_width, j:j + self.config.block_height],
                        self.vector_dimension))
        return np.asarray(image_vectors).astype(float)

    @staticmethod
    def save_image(image, output_image_name):
        imageio.imwrite(output_image_name, image)

    def mse(self, image_b):
        # calculate mean square error between two images
        err = np.sum((self.image.astype(float) - image_b.astype(float)) ** 2)
        err /= float(self.image.shape[0] * self.image.shape[1])
        return err

import sys

import cv2
import numpy as np


class ImageHandler:

    # source image
    image_location = sys.argv[1]
    image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
    image_height = len(image)
    image_width = len(image[0])
    # dimension of the vector
    block_width = int(sys.argv[3])
    block_height = int(sys.argv[4])
    vector_dimension = block_width * block_height

    def split_image_to_blocks(self):
        image_vectors = []
        for i in range(0, self.image_height, self.block_height):
            for j in range(0, ImageHandler.image_width, ImageHandler.block_width):
                image_vectors.append(
                    np.reshape(self.image[i:i + self.block_width, j:j + self.block_height], self.vector_dimension))
        image_vectors = np.asarray(image_vectors).astype(float)
        number_of_image_vectors = image_vectors.shape[0]

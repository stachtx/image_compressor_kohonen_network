from math import log, pow

import numpy as np
import scipy.misc
from scipy.cluster.vq import vq
import imageio
from Config import Config
from ImageHandler import ImageHandler
from KohonnenNetwork import KohonnenNetwork


class Compressor:

    def __init__(self):
        self.compressorConfig = Config().compressorConfig
        self.imageHandlerConfig = Config().imageHandlerConfig
        self.codebook_size = pow(2, self.compressorConfig.bits_per_codevector)

    def compress(self):
        kn_rows = int(pow(2, int((log(self.codebook_size, 2)) / 2)))
        kn_columns = int(self.codebook_size / kn_rows)
        image_handler = ImageHandler()
        image_vectors = image_handler.split_image_to_blocks()
        number_of_image_vectors = image_vectors.shape[0]
        kn = KohonnenNetwork(kn_rows, kn_columns, image_handler.vector_dimension, self.compressorConfig.epochs,
                             number_of_image_vectors, self.compressorConfig.initial_learning_rate,
                             max(kn_rows, kn_columns) / 2)
        reconstruction_values = kn.train(image_vectors)
        image_vector_indices, distance = vq(image_vectors, reconstruction_values)

        image_after_compression = np.zeros([image_handler.image_width, image_handler.image_height], "uint8")
        for index, image_vector in enumerate(image_vectors):
            start_row = int(
                index / (image_handler.image_width / self.imageHandlerConfig.block_width)) * self.imageHandlerConfig.block_height
            end_row = start_row + self.imageHandlerConfig.block_height
            start_column = (index * self.imageHandlerConfig.block_width) % image_handler.image_width
            end_column = start_column + self.imageHandlerConfig.block_width
            image_after_compression[start_row:end_row, start_column:end_column] = \
                np.reshape(reconstruction_values[image_vector_indices[index]],
                           (self.imageHandlerConfig.block_width, self.imageHandlerConfig.block_height))

        output_image_name = "CB_size=" + str(self.codebook_size) + ".png"
        imageio.imwrite(output_image_name, image_after_compression)

        print("Mean Square Error = ", image_handler.mse(image_after_compression))


def main():
    compressor = Compressor()
    compressor.compress()


if __name__ == "__main__":
    main()

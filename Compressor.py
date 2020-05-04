import random
from math import log, pow

import imageio
import numpy as np
from scipy.cluster.vq import vq

from Config import Config
from ImageHandler import ImageHandler
from KohonnenNetwork import KohonnenNetwork


class Compressor:

    def __init__(self):
        self.compressorConfig = Config().compressorConfig
        self.imageHandlerConfig = Config().imageHandlerConfig
        self.codebook_size = pow(2, self.compressorConfig.bits_per_codevector)

    def drawTrainingVectors(self, image_vectors):
        if self.imageHandlerConfig.training_frames <= image_vectors.shape[0]:
            return random.choices(image_vectors.copy(), k=self.imageHandlerConfig.training_frames)
        else:
            return image_vectors

    def compress(self):
        kn_rows = int(pow(2, int((log(self.codebook_size, 2)) / 2)))
        kn_columns = int(self.codebook_size / kn_rows)
        image_handler = ImageHandler()
        is_RGB = image_handler.config.RGB
        if is_RGB:
            image_vectors_R = image_handler.split_image_to_blocks_RGB(2)
            image_vectors_G = image_handler.split_image_to_blocks_RGB(1)
            image_vectors_B = image_handler.split_image_to_blocks_RGB(0)
            number_of_image_vectors = image_vectors_R.shape[0]

            kn = KohonnenNetwork(kn_rows, kn_columns, image_handler.vector_dimension, self.compressorConfig.epochs,
                                 number_of_image_vectors, self.compressorConfig.initial_learning_rate,
                                 max(kn_rows, kn_columns) / 2)

            random_train_vectors_R = self.drawTrainingVectors(image_vectors_R)
            random_train_vectors_G = self.drawTrainingVectors(image_vectors_G)
            random_train_vectors_B = self.drawTrainingVectors(image_vectors_B)
            reconstruction_values_R = kn.train(random_train_vectors_R)
            reconstruction_values_G = kn.train(random_train_vectors_G)
            reconstruction_values_B = kn.train(random_train_vectors_B)
            image_vector_indices_R, distance1 = vq(image_vectors_R, reconstruction_values_R)
            image_vector_indices_G, distance2 = vq(image_vectors_G, reconstruction_values_G)
            image_vector_indices_B, distance3 = vq(image_vectors_B, reconstruction_values_B)
            image_after_compression_RGB = np.zeros([image_handler.image_width, image_handler.image_height, 3], "uint8")

            for index, image_vector in enumerate(image_vectors_R):
                start_row = int(
                    index / (
                            image_handler.image_width / self.imageHandlerConfig.block_width)) * self.imageHandlerConfig.block_height
                end_row = start_row + self.imageHandlerConfig.block_height
                start_column = (index * self.imageHandlerConfig.block_width) % image_handler.image_width
                end_column = start_column + self.imageHandlerConfig.block_width
                for x in range(start_row, end_row, 1):
                    counter = 0
                    for y in range(start_column, end_column, 1):
                        image_after_compression_RGB[x][y][0] = \
                            reconstruction_values_R[image_vector_indices_R[index]][counter]
                        image_after_compression_RGB[x][y][1] = \
                            reconstruction_values_G[image_vector_indices_G[index]][counter]
                        image_after_compression_RGB[x][y][2] = \
                            reconstruction_values_B[image_vector_indices_B[index]][counter]
                        counter = counter + 1

            output_image_name = "RGB_CB_size=" + str(self.codebook_size) + ".png"
            imageio.imwrite(output_image_name, image_after_compression_RGB)
        else:
            image_vectors = image_handler.split_image_to_blocks()
            random_train_vectors = self.drawTrainingVectors(image_vectors)
            number_of_image_vectors = image_vectors.shape[0]

            kn = KohonnenNetwork(kn_rows, kn_columns, image_handler.vector_dimension, self.compressorConfig.epochs,
                                 number_of_image_vectors, self.compressorConfig.initial_learning_rate,
                                 max(kn_rows, kn_columns) / 2)
            reconstruction_values = kn.train(random_train_vectors)
            image_vector_indices, distance = vq(image_vectors, reconstruction_values)

            image_after_compression = np.zeros([image_handler.image_width, image_handler.image_height], "uint8")
            for index, image_vector in enumerate(image_vectors):
                start_row = int(
                    index / (
                                image_handler.image_width / self.imageHandlerConfig.block_width)) * self.imageHandlerConfig.block_height
                end_row = start_row + self.imageHandlerConfig.block_height
                start_column = (index * self.imageHandlerConfig.block_width) % image_handler.image_width
                end_column = start_column + self.imageHandlerConfig.block_width
                image_after_compression[start_row:end_row, start_column:end_column] = \
                    np.reshape(reconstruction_values[image_vector_indices[index]],
                               (self.imageHandlerConfig.block_width, self.imageHandlerConfig.block_height))

            output_image_name = "CB_size=" + str(self.codebook_size) + ".png"
            image_handler.save_image(image_after_compression, output_image_name)
            mse = image_handler.mse(image_after_compression)
            print("Mean Square Error = ", mse)
            print("PSNR measure = ", image_handler.psnr(mse))
            print("Compression ratio = ", image_handler.compression_ratio(output_image_name))


def main():
    compressor = Compressor()
    compressor.compress()


if __name__ == "__main__":
    main()

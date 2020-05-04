from math import exp, pow

import numpy as np
from scipy import spatial


class KohonnenNetwork(object):

    def __init__(self, rows, columns, dimensions, epochs, number_of_input_vectors, alpha, sigma):
        self.rows = rows
        self.columns = columns
        self.dimensions = dimensions
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.number_of_input_vectors = number_of_input_vectors
        self.number_of_iterations = self.epochs * self.number_of_input_vectors

    def get_bmu_location(self, input_vector, weights):

        tree = spatial.KDTree(weights)
        bmu_index = tree.query(input_vector)[1]
        return np.array([int(bmu_index / self.columns), bmu_index % self.columns])

    def update_weights(self, iter_no, bmu_location, input_data):

        learning_rate_op = 1 - (iter_no / float(self.number_of_iterations))
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        distance_from_bmu = []
        for x in range(self.rows):
            for y in range(self.columns):
                distance_from_bmu = np.append(distance_from_bmu, np.linalg.norm(bmu_location - np.array([x, y])))

        neighbourhood_function = [exp(-0.5 * pow(val, 2) / float(pow(sigma_op, 2))) for val in distance_from_bmu]

        final_learning_rate = [alpha_op * val for val in neighbourhood_function]

        for l in range(self.rows * self.columns):
            weight_delta = [val * final_learning_rate[l] for val in (input_data - self.weight_vectors[l])]
            updated_weight = self.weight_vectors[l] + np.array(weight_delta)
            self.weight_vectors[l] = updated_weight

    def draw_and_normalize_wectors(self):
        wectors = np.random.uniform(0, 255, (self.rows * self.columns, self.dimensions))
        for i in range(0, wectors.shape[0]):
            for j in range(0, wectors[i].shape[0]):
                wectors[i][j] = (wectors[i][j] - 0) / (255 - 0)
        self.weight_vectors = wectors

    def train(self, input_data):

        self.draw_and_normalize_wectors()
        iter_no = 0
        for epoch_number in range(self.epochs):
            for index, input_vector in enumerate(input_data):
                bmu_location = self.get_bmu_location(input_vector, self.weight_vectors)
                self.update_weights(iter_no, bmu_location, input_vector)
                iter_no += 1
        return self.weight_vectors

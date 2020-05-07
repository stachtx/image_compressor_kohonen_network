from math import exp, pow

import numpy as np
from scipy import spatial


class KohonnenNetwork(object):

    def __init__(self, number_of_neurons, dimensions, epochs, number_of_input_vectors, alpha, sigma):
        self.dimensions = dimensions
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.number_of_input_vectors = number_of_input_vectors
        self.number_of_iterations = self.epochs * self.number_of_input_vectors
        self.number_of_neurons = number_of_neurons

    def get_bmu_location(self, input_vector, weights):

        tree = spatial.KDTree(weights)
        bmu_index = tree.query(input_vector)[1]
        return np.array([int(bmu_index)])

    def update_weights(self, iter_no, bmu_location, input_data):

        learning_rate_op = 1 - (iter_no / float(self.number_of_iterations))
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        distance_from_bmu = []
        for x in range(self.number_of_neurons):
            distance_from_bmu = np.append(distance_from_bmu, np.linalg.norm(bmu_location - np.array([x])))

        neighbourhood_function = [exp(-0.5 * pow(val, 2) / float(pow(sigma_op, 2))) for val in distance_from_bmu]

        final_learning_rate = [alpha_op * val for val in neighbourhood_function]

        for l in range(self.number_of_neurons):
            weight_delta = [val * final_learning_rate[l] for val in (input_data - self.weight_vectors[l])]
            updated_weight = self.weight_vectors[l] + np.array(weight_delta)
            self.weight_vectors[l] = updated_weight

    def draw_and_normalize_vectors(self):
        vectors = np.random.uniform(0, 255, (self.number_of_neurons, self.dimensions))
        for i in range(0, vectors.shape[0]):
            for j in range(0, vectors[i].shape[0]):
                vectors[i][j] = (vectors[i][j] - 0) / (255 - 0)
        self.weight_vectors = vectors

    def train(self, input_data):

        self.draw_and_normalize_vectors()
        iter_no = 0
        for epoch_number in range(self.epochs):
            for index, input_vector in enumerate(input_data):
                bmu_location = self.get_bmu_location(input_vector, self.weight_vectors)
                self.update_weights(iter_no, bmu_location, input_vector)
                iter_no += 1
        return self.weight_vectors

import numpy as np
from functools import reduce


class VeterbiNode(object):
    def __init__(self):
        self.path = []
        self.prob = 1

    def next_status(self, status, prob):
        ret = VeterbiNode()
        ret.path = self.path[:]
        ret.path.append(status)
        ret.prob = prob
        return ret


class BaseHMM(object):
    def __init__(self, n_status, n_observation):
        self.n_status = n_status
        self.n_observation = n_observation
        self._init_status_vector = None
        self._transform_matrix = None
        self._observation_matrix = None
        self._hidden_status = None

    def _normalize(self, matrix):
        matrix /= np.sum(matrix, axis=1)

    def _init_init_status_vector(self):
        return np.array([1.0 / self.n_status] * self.n_status)

    def _init_transform_matrix(self):
        trans_mat = np.matrix(np.ones((self.n_status, self.n_status)) * 1.0 / self.n_status)
        return trans_mat

    def _init_observation_matrix(self):
        ob_mat = np.matrix(np.ones((self.n_status, self.n_observation)) * 1.0 / self.n_observation)
        return ob_mat

    def predict(self, observations):
        n_time = len(observations)

        # init vertebi node
        veterbi_nodes = [VeterbiNode() for _ in range(self.n_status)]

        # init path info
        for status in range(self.n_status):
            veterbi_nodes[status].path.append(status)
            veterbi_nodes[status].prob = self._init_status_vector[status] * self._observation_matrix[status, observations[0]]

        # iteratively calculate path
        for time in range(1, n_time):
            new_veterbi_nodes = []
            for status in range(self.n_status):
                probs = [veterbi_nodes[prev_status].prob * self._transform_matrix[prev_status, status] *
                         self._observation_matrix[status, observations[time]] for prev_status in range(self.n_status)]
                index = np.where(probs == np.max(probs))[0][0]
                new_veterbi_nodes.append(veterbi_nodes[index].next_status(status, probs[index]))

            veterbi_nodes = new_veterbi_nodes
            max_prob = reduce(lambda node1, node2: node1 if node1.prob > node2.prob else node2, veterbi_nodes).prob
            if max_prob < 1:
                for node in veterbi_nodes:
                    node.prob *= 2 / max_prob

        # find max path
        best_path_node = reduce(lambda node1, node2: node1 if node1.prob >= node2.prob else node2, veterbi_nodes)

        return best_path_node.path

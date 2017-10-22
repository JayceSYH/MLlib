from HMM.BaseHMM import BaseHMM
import numpy as np


class SupervisedHMM(BaseHMM):
    def __init__(self, n_status, n_observation):
        super(SupervisedHMM, self).__init__(n_status, n_observation)

    def fit(self, hidden_status, observations, laplace_smooth=False):
        # init model
        self._init_status_vector = np.array([0.0] * self.n_status)
        self._transform_matrix = np.matrix(np.zeros((self.n_status, self.n_status)))
        self._observation_matrix = np.matrix(np.zeros((self.n_status, self.n_observation)))

        assert len(hidden_status) == len(observations), \
            "hidden status length({0}) and observation length({1}) not matched".\
                format(len(hidden_status), len(observations))
        n_time = len(hidden_status)

        for time in range(n_time):
            if time != n_time - 1:
                self._transform_matrix[hidden_status[time], hidden_status[time + 1]] += 1

            self._observation_matrix[hidden_status[time], observations[time]] += 1

            self._init_status_vector[hidden_status[time]] += 1

        # laplace smooth
        if laplace_smooth:
            self._transform_matrix += 1
            self._observation_matrix += 1
            self._init_status_vector += 1

        # normalize
        self._normalize(self._transform_matrix)
        self._normalize(self._observation_matrix)
        self._init_status_vector /= np.sum(self._init_status_vector)


if __name__ == "__main__":
    hmm = SupervisedHMM(3, 2)

    observations = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1] * 100
    hidden_status = [0, 1, 1, 2, 0, 2, 0, 0, 2, 1] * 100

    hmm.fit(hidden_status, observations)

    print(hmm._init_status_vector)
    print(hmm._transform_matrix)
    print(hmm._observation_matrix)

    print(observations)
    print(hmm.predict(observations))
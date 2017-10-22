import numpy as np
from HMM.HMMComm import HMMComm
from HMM.BaseHMM import BaseHMM


class UnsupervisedHMM(BaseHMM):
    def __init__(self, n_status, n_observation):
        super(UnsupervisedHMM, self).__init__(n_status, n_observation)

    def fit(self, observations, n_steps=10, init_status=None, init_transformation_matrix=None,
            init_observation_matrix=None):

        # init model params
        if init_status is not None:
            self._init_status_vector = np.array(init_status)
        else:
            self._init_status_vector = self._init_init_status_vector()

        if init_transformation_matrix is not None:
            self._transform_matrix = init_transformation_matrix
        else:
            self._transform_matrix = self._init_transform_matrix()

        if init_observation_matrix is not None:
            self._observation_matrix = init_observation_matrix
        else:
            self._observation_matrix = self._init_observation_matrix()

        # n times
        n_times = len(observations)

        # iteratively calculate parameters
        for step in range(n_steps):
            # calculate forward/backward probability
            forward = HMMComm.calculate_forward_probability(observations, self._init_status_vector,
                                                            self._transform_matrix, self._observation_matrix)
            backward = HMMComm.calculate_backward_probability(observations, self._init_status_vector,
                                                              self._transform_matrix, self._observation_matrix)

            # calculate gamma and epsilon
            status_i_condition_observations = np.zeros((self.n_status, n_times))
            for time in range(n_times):
                for i in range(self.n_status):
                    status_i_condition_observations[i, time] = HMMComm.\
                        calculate_status_i_probability_condition_observation(time, i, forward, backward)

            status_i2j_condition_observations = np.zeros((self.n_status, self.n_status, n_times - 1))
            for time in range(n_times - 1):
                for i in range(self.n_status):
                    for j in range(self.n_status):
                        status_i2j_condition_observations[i,j,time] = \
                            HMMComm.calculate_status_i2j_probability_condition_observation\
                                (time, i, j, forward, backward, self._transform_matrix, self._observation_matrix,
                                 observations)

            # calculate init status matrix
            for status in range(self.n_status):
                self._init_status_vector[status] = status_i_condition_observations[status, 0]
            self._init_status_vector /= np.sum(self._init_status_vector)

            # calculate transformation matrix
            for i in range(self.n_status):
                for j in range(self.n_status):
                    top = 0
                    down = 0
                    for time in range(n_times - 1):
                        top += status_i2j_condition_observations[i, j, time]
                        down += status_i_condition_observations[i, time]
                    self._transform_matrix[i, j] = top / down
            self._normalize(self._transform_matrix)

            # calculate observation matrix
            for status in range(self.n_status):
                for observation in range(self.n_observation):
                    top = 0
                    down = 0
                    for time in range(n_times):
                        if observations[time] == observation:
                            top += status_i_condition_observations[status, time]
                        down += status_i_condition_observations[status, time]
                    self._observation_matrix[status, observation] = top / down
            self._normalize(self._observation_matrix)


if __name__ == "__main__":
    hmm = UnsupervisedHMM(3, 2)
    hmm.fit([0, 1, 1, 0, 0, 0, 1, 0, 1], n_steps=50)

    print(hmm._init_status_vector)
    print(hmm._transform_matrix)
    print(hmm._observation_matrix)

    hmm.predict([0, 1, 1, 0, 0, 0, 1, 0, 1])
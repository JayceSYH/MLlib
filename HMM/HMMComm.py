import numpy as np
from functools import reduce


class HMMComm(object):
    @staticmethod
    def calculate_forward_probability(observations, init_status_vector, status_transformation_matrix,
                                      observation_matrix):
        n_status = observation_matrix.shape[0]
        n_time = len(observations)

        # probability matrix with n_status rows and n_observation columns
        forward_prob_matrix = np.matrix(np.zeros((n_status, n_time)))

        # initialize status
        status_vector = init_status_vector

        # iteratively calculate probability
        for i in range(len(observations)):
            # calculate forward status probability
            joint_status_vector = (status_vector * observation_matrix[:, observations[i]].getA1())
            forward_prob_matrix[:, i] = joint_status_vector.reshape(-1, 1)

            # calculate next status vector
            status_vector = (joint_status_vector * status_transformation_matrix).getA1()

        return forward_prob_matrix

    @staticmethod
    def calculate_backward_probability(observations, init_status_vector, status_transformation_matrix,
                                       observation_matrix):
        n_status = observation_matrix.shape[0]
        n_time = len(observations)

        # probability matrix with n_status rows and n_observation columns
        backward_prob_matrix = np.matrix(np.zeros((n_status, n_time)))

        # init backward final status
        backward_prob_matrix[:, -1] = 1

        # iteratively calculate probability
        for i in range(len(observations) - 2, -1, -1):
            # calculate backward status probability
            backward_prob_matrix[:, i] = ((status_transformation_matrix *
                                          observation_matrix[:, observations[i]]).getA1() *
                                          backward_prob_matrix[:, i + 1].getA1()).reshape(-1, 1)

        return backward_prob_matrix

    @staticmethod
    def calculate_status_i_probability_joint_observations(time, status, forward, backward):
        return forward[status, time] * backward[status, time]

    @staticmethod
    def calculate_status_i_probability_condition_observation(time, status, forward, backward):
        # n_status
        n_status = forward.shape[0]

        # partial on status
        partial_func = lambda x_status: \
            HMMComm.calculate_status_i_probability_joint_observations(time, x_status, forward, backward)

        # normalization term
        normalization_term = reduce(lambda x, y: x + y,
                                    map(partial_func, range(n_status)))

        # conditional probability
        return HMMComm.calculate_status_i_probability_joint_observations(time, status, forward, backward) / normalization_term

    @staticmethod
    def calculate_status_i2j_probability_joint_observations(time, status_t, status_tp1, forward, backward,
                                                            transformation_matrix, observation_matrix, observations):
        return forward[status_t, time] * transformation_matrix[status_t, status_tp1] * \
               observation_matrix[status_tp1, observations[time + 1]] * backward[status_tp1, time + 1]

    @staticmethod
    def calculate_status_i2j_probability_condition_observation(time, status_t, status_tp1, forward, backward,
                                                               transformation_matrix, observation_matrix, observations):
        n_status = observation_matrix.shape[0]

        # partial on status_t and status_tp1
        partial_func = lambda x_status_t, x_status_tp1: \
            HMMComm.calculate_status_i2j_probability_joint_observations(time, x_status_t, x_status_tp1, forward,
                                                                        backward, transformation_matrix,
                                                                        observation_matrix, observations)

        # normalization term
        normalization_term = 0

        for i in range(n_status):
            inner_partial_func = lambda x_status_tp1: partial_func(i, x_status_tp1)
            normalization_term += reduce(lambda x, y: x + y,
                                         map(inner_partial_func, range(n_status)))

        # conditional probability
        return HMMComm.calculate_status_i2j_probability_joint_observations(time, status_t, status_tp1, forward,
                                                                           backward, transformation_matrix,
                                                                           observation_matrix, observations) / normalization_term

    @staticmethod
    def calculate_joint_probability_by_forward(forward):
        return np.sum(forward[:, -1])

    @staticmethod
    def calculate_joint_probability_by_backward(backward):
        return np.sum(init_status * ob_matrix[:, observations[0]].getA1() * backward[:, 0].getA1())


if __name__ == "__main__":
    init_status = np.array([1.0, 0, 0])
    trans_mt = np.matrix([
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
    ])

    ob_matrix = np.matrix([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ])

    observations = [0, 1, 1, 0, 0, 0, 1, 0, 1]

    forward = (HMMComm.calculate_forward_probability(observations, init_status, trans_mt, ob_matrix))
    print(forward)
    print(np.sum(forward[:, -1]))

    backward = HMMComm.calculate_backward_probability(observations, init_status, trans_mt, ob_matrix)
    print(backward)
    print(np.sum(init_status * ob_matrix[:, observations[0]].getA1() * backward[:, 0].getA1()))

    print("gamma(time:1 status:0):{}".
          format(HMMComm.calculate_status_i_probability_condition_observation(1, 0, forward, backward)))
    print("gamma(time:1 status:1):{}".
          format(HMMComm.calculate_status_i_probability_condition_observation(1, 1, forward, backward)))
    print("gamma(time:1 status:2):{}".
          format(HMMComm.calculate_status_i_probability_condition_observation(1, 2, forward, backward)))

    print("epsilon(time:1 status:1 time:2 status:0):{}".
          format(HMMComm.calculate_status_i2j_probability_condition_observation(1, 1, 0, forward, backward, trans_mt,
                                                                                ob_matrix, observations)))
    print("epsilon(time:1 status:1 time:2 status:1):{}".
          format(HMMComm.calculate_status_i2j_probability_condition_observation(1, 1, 1, forward, backward, trans_mt,
                                                                                ob_matrix, observations)))
    print("epsilon(time:1 status:1 time:2 status:2):{}".
          format(HMMComm.calculate_status_i2j_probability_condition_observation(1, 1, 2, forward, backward, trans_mt,
                                                                                ob_matrix, observations)))

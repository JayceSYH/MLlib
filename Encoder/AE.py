import tensorflow as tf
from functools import reduce
import math


class AE(object):
    def __init__(self, input_size, output_size, activation="tanh", loss="mse",
                 optimizer="gd", p=0.2, beta=1.0, C=1.0, learning_rate=0.01):
        # net structure
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._loss_func = loss
        self._p = p
        self._beta = beta
        self._C = C
        self._optimizer = optimizer
        self._learning_rate = learning_rate

        # input/output nodes
        self._encoder_input = None
        self._encoder_output = None

        self._decoder_input = None
        self._decoder_output = None

        # KL
        self._KL = None

        # regularization
        self._L1 = None
        self._L2 = None

        # param nodes
        self._encoder_w = tf.Variable(tf.random_normal((input_size, output_size)))
        self._decoder_w = tf.Variable(tf.random_normal((output_size, input_size)))
        self._decoder_linear_transform_w = tf.Variable(tf.random_normal((input_size, input_size)))
        self._encoder_offset = tf.Variable(0.0)
        self._decoder_offset = tf.Variable(0.0)
        self._decoder_linear_transform_offset = tf.Variable(0.0)

        # loss
        self._loss = None

        # train process
        self._train_process = None

    def encoder(self, prev_layer):
        _input = prev_layer

        a_out = tf.matmul(_input, self._encoder_w) + self._encoder_offset

        # activation
        if self._activation == "tanh":
            self._encoder_output = tf.tanh(a_out)
        elif self._activation == "sigmoid":
            self._encoder_output = tf.sigmoid(a_out)
        else:
            assert False, "Invalid activation {}".format(self._activation)

        # regularization
        self._L1 = self._C * tf.reduce_sum(tf.abs(self._encoder_w))
        self._L2 = self._C * tf.reduce_sum(tf.square(self._encoder_w))

        # KL
        if self._activation == "tanh":
            p_act = (tf.reduce_mean(self._encoder_output) + 1) / 2
        elif self._activation == "sigmoid":
            p_act = tf.reduce_mean(self._encoder_output)
        else:
            assert False, "Invalid activation {}".format(self._activation)

        self._KL = self._p * tf.log(self._p / p_act) + (1 - self._p) * tf.log((1 - self._p) / (1 - p_act))

        return self._encoder_output

    def encoder_as_input(self):
        self._encoder_input = tf.placeholder(tf.float32, (None, self._input_size))
        return self._encoder_input, self.encoder(self._encoder_input)

    def decoder(self, prev_layer):
        _input = prev_layer

        a_out = tf.matmul(_input, self._decoder_w) + self._decoder_offset

        # activation
        if self._activation == "tanh":
            activation_output = tf.tanh(a_out)
        elif self._activation == "sigmoid":
            activation_output = tf.sigmoid(a_out)
        else:
            assert False, "Invalid activation {}".format(self._activation)

        # linear transform
        self._decoder_output = tf.matmul(activation_output, self._decoder_linear_transform_w) + \
                               self._decoder_linear_transform_offset

        return self._decoder_output

    def decoder_as_output(self, prev_layer, origin_input, kl_list, lxs=None, total_weight_num=None):
        self.decoder(prev_layer)

        # calculate mse loss
        if self._loss_func == "mse":
            loss = tf.reduce_mean(tf.square(self._decoder_output - origin_input))
        else:
            assert False, "only mse is currently implemented"

        # calculate regularization loss
        if lxs:
            assert total_weight_num is not None, "total number of weight node must be set"
            regularization_loss = tf.reduce_sum(lxs) / total_weight_num
        else:
            regularization_loss = 0.0

        # calculate total loss
        self._loss = loss + self._beta * tf.reduce_sum(kl_list) + regularization_loss

        # optimize
        if self._optimizer == "gd":
            opt = tf.train.GradientDescentOptimizer(self._learning_rate)
        elif self._optimizer == "adam":
            opt= tf.train.AdamOptimizer()
        else:
            opt = self._optimizer

        # train process
        self._train_process = opt.minimize(self._loss)

        return self._train_process, self._decoder_output

    @property
    def KL(self):
        return self._KL

    @property
    def L1(self):
        return self._L1

    @property
    def L2(self):
        return self._L2

    @property
    def shape(self):
        return self._input_size, self._output_size

    @property
    def weight_size(self):
        return self._input_size * self._output_size

    @property
    def loss(self):
        return self._loss


if __name__ == "__main__":
    import numpy as np

    ae = AE(10, 5, beta=1, optimizer="adam", activation="sigmoid")
    print("build encoder")
    en_in, en_out = ae.encoder_as_input()
    print("build decoder")
    train, de_out = ae.decoder_as_output(en_out, en_in, [ae.KL], lxs=[ae.L2], total_weight_num=ae.weight_size)

    X_train = np.matrix([([2 * i] * 10) for i in range(20)])
    print(X_train)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for i in range(20000):
            sess.run([train], {
                en_in: X_train
            })

            print(sess.run([ae._loss], {
                en_in: X_train
            }))

        out = sess.run([de_out], {
            en_in: X_train
        })

        print(out)

        print(sess.run([en_out], {
            en_in:X_train
        }))


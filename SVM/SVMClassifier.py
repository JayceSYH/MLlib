import tensorflow as tf
import numpy as np


class SVMClassifier(object):
    def __init__(self, kernel="linear", alpha=0.01, C=1):
        self.kernel = kernel
        self.alpha = alpha
        self.C = C

        # param
        self.n_feature = None
        self.batch_size = None

        # tensor
        self.t_X = None
        self.t_y = None
        self.t_w = None
        self.t_b = None
        self.pred = None
        self.loss = None
        self.train_process = None
        self.sess = tf.Session()

        # result
        self.w = None
        self.b = None

    def _hinge_loss(self, w, b, x, y):
        # calculate loss part
        function_distance = (tf.matmul(x, w) + b) * y
        loss = tf.reduce_sum(tf.maximum(0.0, 1 - function_distance)) / self.batch_size

        # regularization
        regularization = 1 / (2 * self.C) * tf.reduce_sum(w * w)

        # return sum_loss, loss_part

        # calculate normalization part
        # w_mul = tf.transpose(w) * w
        # regularize = 1 / (2 * self.C) * tf.reduce_sum(w_mul)
        return loss + regularization

    def _train(self, x, y):
        w = tf.Variable(tf.random_normal((self.n_feature, 1)))
        # w = tf.Variable([[0.0]] * self.n_feature)
        b = tf.Variable(0.0)
        self.loss = self._hinge_loss(w, b, x, y)
        opt = tf.train.AdamOptimizer(self.loss)
        self.train_process = opt.minimize(self.loss)
        pred = tf.matmul(x, w) + b
        self.pred = tf.transpose(pred / tf.abs(pred))
        self.t_w = w
        self.t_b = b

    def fit(self, X, y):
        y = np.reshape(y, (-1, 1))

        # init param
        self.n_feature = X.shape[1]
        self.batch_size = X.shape[0]
        self.t_X = tf.placeholder(tf.float32, (None, self.n_feature))
        self.t_y = tf.placeholder(tf.float32, (None, 1))

        # make train graph
        self._train(self.t_X, self.t_y)

        # initialize variable
        self.sess.run(tf.initialize_all_variables())

        # train
        for i in range(40):
            _, loss, w, b = self.sess.run([self.train_process, self.loss, tf.transpose(self.t_w), self.t_b], feed_dict={self.t_X: X, self.t_y:y})
            self.w = w[0]
            self.b = b

            print("w:{}".format(w))

    def predict(self, X):
        return self.sess.run(self.pred, {self.t_X:X})[0]
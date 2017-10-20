import tensorflow as tf
from Encoder.AE import AE


class SAE(object):
    def __init__(self, ae_list):
        # net structure
        self._input_size = ae_list[0].shape[0]
        self._output_size = ae_list[-1].shape[0]
        self._ae_list = ae_list

        # input/output
        self._encoder_input = None
        self._encoder_output = None
        self._decoder_input = None
        self._decoder_output = None

        # tensor node
        self._train_process = None
        self._loss = None

        # KL
        self._KL = None

    def encoder(self, input_):
        next_input = None
        total_input = input_

        # encode progress
        for i, ae in enumerate(self._ae_list):
            if i == 0:
                next_input = ae.encoder(total_input)
            else:
                next_input = ae.encoder(next_input)

        # KL
        self._KL = [ae.KL for ae in self._ae_list]

        self._encoder_output = next_input
        return next_input

    def encoder_as_input(self):
        self._encoder_input = tf.placeholder(tf.float32, (None, self._input_size))
        return self._encoder_input, self.encoder(self._encoder_input)

    def decoder(self, input_, origin_input):
        next_input = input_
        last_no = len(self._ae_list) - 1
        train_process = None

        # decode progress
        for i, ae in enumerate(reversed(self._ae_list)):
            if i == last_no:
                train_process, next_input = ae.decoder_as_output(next_input, origin_input, self._KL)
                self._loss = ae.loss
            else:
                next_input = ae.decoder(next_input)

        self._train_process, self._decoder_output = train_process, next_input
        return train_process, next_input

    def train(self, X, n_steps=50):
        input_, output_ = self.encoder_as_input()
        train_process_, _ = self.decoder(output_, input_)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in range(n_steps):
                sess.run([train_process_], {
                    input_: X
                })

            loss, out = sess.run([self._loss, self._decoder_output], {
                input_:X
            })
            print("training done, loss:{}".format(loss))
            print("output:\n{}".format(out))

    @property
    def encoder_input(self):
        return self._encoder_output

    @property
    def encoder_output(self):
        return self._encoder_output

    @property
    def decoder_output(self):
        return self._decoder_output

    @property
    def loss(self):
        return self._loss

    @staticmethod
    def auto_stack(ae_list, X, n_steps=1000):
        for i in range(len(ae_list)):
            print("total {0} layers, training layer {1}".format(len(ae_list), i + 1))
            train_sae = SAE(ae_list[:i + 1])
            train_sae.train(X, n_steps=n_steps)

            if i == len(ae_list) - 1:
                return train_sae



if __name__ == "__main__":
    import numpy as np

    X_train = np.matrix([([2 * i] * 10) for i in range(20)])
    print(X_train)

    ae_list = [AE(10, 20, beta=1, optimizer="adam", activation="sigmoid"),
               AE(20, 15, beta=1, optimizer="adam", activation="sigmoid"),
               AE(15, 10, beta=1, optimizer="adam", activation="sigmoid"),
               AE(10, 8, beta=1, optimizer="adam", activation="sigmoid")]

    # train
    sae = SAE.auto_stack(ae_list, X_train, n_steps=20000)



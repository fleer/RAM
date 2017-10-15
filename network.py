import keras
from keras import backend as K
from keras import layers
import numpy as np

class RAM():

    loc_std = 0.11
    glimpses = 6
    def __init__(self, totalSensorBandwidth, batch_size, glimpses):

        self.discounted_r = np.zeros((batch_size, 1))
        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses
        self.p_loc = np.zeros((batch_size, glimpses * 2), dtype='float32')
        self.sampled_locs =[] #np.zeros((batch_size, glimpses, 2))
        self.mean_locs = [] #np.zeros((batch_size, glimpses, 2))
        self.big_net()

      #  self.ram = self.core_network()
      #  self.gl_net = self.glimpse_network()
      #  self.act_net = self.action_network()
      #  self.loc_net = self.location_network()



    # to use for maximum likelihood with glimpse location
    def gaussian_pdf(self, mean, sample):
        Z = 1.0 / (self.loc_std * np.sqrt(2.0 * np.math.pi))
        a = -np.square(np.asarray(sample) - np.asarray(mean)) / (2.0 * np.square(self.loc_std))
        return Z * np.exp(a)

    def REINFORCE_loss(self, action_p):
        def rloss(y_true, y_pred):
            max_p_y = K.argmax(action_p, axis=-1)
            R = K.equal(max_p_y, K.cast(y_true, 'int64')) # reward per example
            R = K.reshape(R, (self.batch_size, 1))
            R = K.cast(R, 'float32')
            #log_l = np.concatenate([K.log(action_p + 1e-5) * y_true + K.log(self.p_loc + 1e-5) * R], axis=1)
            #log_l = K.concatenate([K.log(action_p + 1e-5) * y_true, K.log(self.p_loc + 1e-5) * R], axis=1)
            #loss = K.sum(log_l,axis=-1)
            log_l =  K.log(K.cast(self.p_loc, 'float32') + 1e-5) * R
            loss = K.mean(log_l, axis=-1) + K.sum(y_pred-y_pred, axis= -1)
            return -loss
            #return K.mean(K.sum(y_pred - y_pred, axis=-1) + K.sum(- log_l, axis=-1), axis=0)
        return rloss

    def big_net(self):
        glimpse_model_i = keras.layers.Input(batch_shape=(self.batch_size, self.totalSensorBandwidth),
                                             name='glimpse_input')
        glimpse_model = keras.layers.Dense(128, activation='relu')(glimpse_model_i)
        glimpse_model_out = keras.layers.Dense(256)(glimpse_model)

        location_model_i = keras.layers.Input(batch_shape=(self.batch_size, 2),
                                              name='location_input')
        location_model = keras.layers.Dense(128,
                                            activation = 'relu'
                                            )(location_model_i)
        location_model_out = keras.layers.Dense(256)(location_model)

        model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        glimpse_network_output = keras.layers.Dense(256,
                                                    activation='relu')(model_merge)
        rnn_input = keras.layers.Reshape([256,1])(glimpse_network_output)
        model = keras.layers.recurrent.LSTM(256,recurrent_initializer="he_uniform", activation='relu',return_sequences=False, stateful=True, unroll=True, name = 'rnn')(rnn_input)

        #    model = keras.layers.recurrent.SimpleRNN(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
    #                                     recurrent_initializer='orthogonal', bias_initializer='zeros',
    #                                     kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    #                                     activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    #                                     bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
    #                                     return_sequences=False, return_state=False, go_backwards=False, stateful=True,
    #                                     unroll=False, name = 'rnn')(rnn_input)
        action_out = keras.layers.Dense(10,
                                 activation='softmax',
                                 name='action_output'
                                 )(model)
        location_out = keras.layers.Dense(2,
                                 activation='tanh',
                                 name='location_output'
                                 )(model)

        self.ram = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out])
        #self.ram.compile(optimizer=keras.optimizers.adam(lr=0.001),
        self.ram.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False),
                         loss={'action_output': 'categorical_crossentropy',
                               'location_output': self.REINFORCE_loss(action_p=action_out)})
        #self.ram.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False),
        #self.ram.compile(optimizer=keras.optimizers.adam(lr=0.01),
        #                 loss={'action_output': 'categorical_crossentropy', 'location_output': self.dummy_loss})

        #self.ram_loc = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out])
        #self.ram_loc.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False),
        #self.ram_loc.compile(optimizer=keras.optimizers.adam(lr=0.01),
        #                 loss={'action_output': self.dummy_loss, 'location_output': self.REINFORCE_loss(action_p=action_out)})

        #self.ram_loc.set_weights(self.ram.get_weights())



    def cat_ent(self, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)

    def dummy_loss(self, y_true, y_pred):
        return K.sum(y_pred-y_pred, axis=-1)


    def train(self, zooms, loc_input, true_a):
        #self.discounted_r = np.zeros((self.batch_size, 1))
        #for b in range(self.batch_size):
        #    self.discounted_r[b] = np.sum(self.compute_discounted_R(r[b], .99))
        self.p_loc = self.gaussian_pdf(self.mean_locs, self.sampled_locs)
        self.p_loc = np.reshape(self.p_loc, (self.batch_size, self.glimpses * 2))
        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))
      #  loc_input = np.reshape(loc, (self.batch_size, 2))
        ath = self.dense_to_one_hot(true_a)
        #self.ram.fit({'glimpse_input': glimpse_input, 'location_input': loc_input},
        #                        {'action_output': ath, 'location_output': ath}, epochs=1, batch_size=self.batch_size, verbose=1, shuffle=False)

       # self.ram_loc.set_weights(self.ram.get_weights())
        self.ram.fit({'glimpse_input': glimpse_input, 'location_input': loc_input},
                                {'action_output': ath, 'location_output': ath}, epochs=1, batch_size=self.batch_size, verbose=2, shuffle=False)
       # self.ram_loc.fit({'glimpse_input': glimpse_input, 'location_input': loc_input},
       #              {'action_output': ath, 'location_output': ath}, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=False)
       # self.ram.get_layer('location_output').set_weights(self.ram_loc.get_layer('location_output').get_weights())

    def reset_states(self):
        self.ram.reset_states()
      #  self.ram.get_layer('rnn').reset_states()
      #  self.ram_loc.get_layer('rnn').reset_states()

    def compute_discounted_R(self, R, discount_rate=.99):
        """Returns discounted rewards
        Args:
            R (1-D array): a list of `reward` at each time step
            discount_rate (float): Will discount the future value by this rate
        Returns:
            discounted_r (1-D array): same shape as input `R`
                but the values are discounted
        Examples:
            >>> R = [1, 1, 1]
            >>> self.compute_discounted_R(R, .99) # before normalization
            [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
        """
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add

        #discounted_r -= discounted_r.mean() / discounted_r.std()

        return discounted_r



    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        # copied from TensorFlow tutorial
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def choose_action(self,X,loc):

        glimpse_input = np.reshape(X, (self.batch_size, self.totalSensorBandwidth))
       # self.ram_loc.predict_on_batch({"glimpse_input": glimpse_input, 'location_input': loc})
        return self.ram.predict_on_batch({"glimpse_input": glimpse_input, 'location_input': loc})
        #gl_out = self.gl_net.predict_on_batch([glimpse_input, loc])
        #ram_out = self.ram.predict_on_batch(np.reshape(gl_out, (self.batch_size, 1, 256)))
       # print self.act_net.predict_on_batch(ram_out)
       # print self.loc_net.predict_on_batch(ram_out)

        #return self.act_net.predict_on_batch(ram_out), self.loc_net.predict_on_batch(ram_out), ram_out, gl_out


    def get_best_action(self,prob_a):
        return np.argmax(prob_a)

def main():
    totalSensorBandwidth = 3 * 8 * 8 * 1
    ram = RAM(totalSensorBandwidth, 32)


if __name__ == '__main__':
    main()


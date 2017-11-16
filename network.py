import keras
from keras import backend as K
from keras import layers
import numpy as np
import os
from collections import defaultdict

class RAM():

    glimpses = 6
    def __init__(self, totalSensorBandwidth, batch_size, glimpses, optimizer,
                 lr, lr_decay, min_lr, momentum, loc_std, clipnorm, clipvalue):

        # TODO --> Integrate Discount Factor for Reward
        self.discounted_r = np.zeros((batch_size, 1))
        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.lr = lr
        self.loc_std = loc_std
        # Learning Rate Decay
        if self.lr_decay != 0:
            self.lr_decay_rate = ((lr - min_lr) /
                                 lr_decay)

        self.big_net(lr, momentum)


    def big_net(self, lr, momentum):
        glimpse_model_i = keras.layers.Input(batch_shape=(self.batch_size, self.totalSensorBandwidth),
                                             name='glimpse_input')
        glimpse_model = keras.layers.Dense(128, activation='relu',
                                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           name='glimpse_2'
                                           )(glimpse_model_i)

        glimpse_model_out = keras.layers.Dense(256,
                                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                               name='glimpse_3'
                                               )(glimpse_model)

        location_model_i = keras.layers.Input(batch_shape=(self.batch_size, 2),
                                              name='location_input')

        location_model = keras.layers.Dense(128,
                                            activation = 'relu',
                                            kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            name='location_1'
                                            )(location_model_i)

        location_model_out = keras.layers.Dense(256,
                                            kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                name='location_2'
                                                )(location_model)

        model_merge = keras.layers.add([glimpse_model_out, location_model_out], name='add')
        glimpse_network_output  = keras.layers.Lambda(lambda x: K.relu(x))(model_merge)

        rnn_input = keras.layers.Reshape((256,1))(glimpse_network_output)
        model_output = keras.layers.recurrent.SimpleRNN(256,recurrent_initializer="zeros", activation='relu',
                                                return_sequences=False, stateful=True, unroll=True,
                                                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                name = 'rnn')(rnn_input)

        action_out = keras.layers.Dense(10,
                                 activation='softmax',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='action_output',
                                 )(model_output)
        location_out = keras.layers.Dense(2,
                                 activation='tanh',
                                 #kernel_initializer=keras.initializers.glorot_uniform(),
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                # bias_initializer=keras.initializers.glorot_uniform(),
                                 name='location_output',
                                 )(model_output)
        baseline_output = keras.layers.Dense(1,
                                 activation='sigmoid',
                               #  kernel_initializer=keras.initializers.glorot_uniform(),
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='baseline_output',
                                         )(model_output)

        self.ram = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out, baseline_output])
        self.ram.compile(optimizer=keras.optimizers.SGD(lr=lr , momentum=momentum, decay=0.0, nesterov=False),
                         loss={'action_output': self.CROSS_ENTROPY,
                               'location_output': self.REINFORCE_LOSS(action_p=action_out, baseline=baseline_output),
                               'baseline_output': self.BASELINE_LOSS(action_p=action_out)})

    def CROSS_ENTROPY(self, y_true, y_pred):
      #  self.ram.trainable = True
        return K.categorical_crossentropy(y_true, y_pred)

    def REINFORCE_LOSS(self, action_p, baseline):
        def loss(y_true, y_pred):
            max_p_y = K.argmax(action_p)
            action = K.argmax(y_true)

            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R = K.cast(R, 'float32')
            R_out = K.reshape(R, (self.batch_size,1))

            # Individual loss for location network
            # Compute loss via REINFORCE algorithm
            # for gaussian distribution
            # d ln(f(m,s,x))   (x - m)
            # -------------- = -------- with m = mean, x = sample, s = standard_deviation
            #       d m          s**2

            sample_loc = K.random_normal(y_pred.shape, y_pred, self.loc_std)
            #sample_loc = K.tanh(sample_loc)
            #TODO: Check how to deal with the 2 dims (x,y) of location
            R = K.tile(R_out, [1, 2])
            b = K.tile(baseline, [1, 2])
            loss_loc = ((sample_loc - y_pred)/(self.loc_std*self.loc_std)) * (R -b)
            return - loss_loc
      #  self.ram.trainable = False
      #  self.ram.get_layer('location_output').trainable = True
        return loss

    def BASELINE_LOSS(self, action_p):
        def loss(y_true, y_pred):
            max_p_y = K.argmax(action_p)
            action = K.argmax(y_true)

            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R = K.cast(R, 'float32')
            R_out = K.reshape(R, (self.batch_size,1))
            return K.mean(K.square(R_out - y_pred), axis=-1)
      #  self.ram.trainable = False
      #  self.ram.get_layer('baseline_output').trainable = True
        return loss


    def learning_rate_decay(self):
        lr = K.get_value(self.ram.optimizer.lr)
        # Linear Learning Rate Decay
        lr = max(self.min_lr, lr - self.lr_decay_rate)
        K.set_value(self.ram.optimizer.lr, lr)
        return lr

    def train(self, zooms, loc_input, true_a):
      #  self.ram.trainable = True
        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))

        loss = self.ram.train_on_batch({'glimpse_input': glimpse_input, 'location_input': loc_input},
                                       {'action_output': true_a, 'location_output': true_a, 'baseline_output': true_a})

        #if self.lr_decay !=0:
        #   lr = self.learning_rate_decay()
        #else:
        #   lr = self.lr
        #self.rnn.set_weights(new_weights)
        #ath = keras.utils.to_categorical(true_a, self.output_dim)
        #self.ram.fit({'glimpse_input': glimpse_input, 'location_input': loc_input},
        #                        {'action_output': ath, 'location_output': ath}, epochs=1, batch_size=self.batch_size, verbose=1, shuffle=False)
       # print "--------------------------------------"
       # print loss
       # print loss_l
       # print loss_b

        #return loss, loss_l, loss_b, R#, np.mean(b, axis=-1)
        return np.mean(loss)

    def reset_states(self):
        self.ram.reset_states()

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
        print R
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add

        discounted_r -= discounted_r.mean() / discounted_r.std()

        return discounted_r

    def choose_action(self,X,loc):

        glimpse_input = np.reshape(X, (self.batch_size, self.totalSensorBandwidth))
        action_prob, loc, _ = self.ram.predict_on_batch({"glimpse_input": glimpse_input, 'location_input': loc})
        #return self.act_net.predict_on_batch(ram_out), self.loc_net.predict_on_batch(ram_out), ram_out, gl_out
        return action_prob, loc

    def save_model(self, path, filename):
        """
        Saves the model to ``model.json`` file and
        also the model weights to model.h5 file
        """
        model_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        # serialize model to JSON
        model_json = self.ram.to_json()
        with open(model_fn + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.ram.save_weights(model_fn + ".h5")

    def load_model(self, path, filename):
        """
        Saves the model to ``model.json`` file and
        also the model weights to model.h5 file
        """
        model_fn = os.path.join(path, filename)
        if os.path.isfile(model_fn):
            # load json and create model
            json_file = open(model_fn + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.ram = keras.models.model_from_json(loaded_model_json)
            # load weights into new model
            self.ram.load_weights(model_fn + ".h5")
            return True
        else:
            return False


def main():
    totalSensorBandwidth = 3 * 8 * 8 * 1
    ram = RAM(totalSensorBandwidth, 32, 6, "sdg", 0.001, 20, 0.0001, 0.9, 0.11, 1, 1)
    ram.save_model("./", "test")
    print "Model saved..."
    ram.load_model("./", "test")
    print "Model loaded..."


if __name__ == '__main__':
    main()


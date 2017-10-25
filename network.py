import keras
from keras import backend as K
from keras import layers
import numpy as np
# from RWA import RWA

class RAM():

    loc_std = 0.11
    glimpses = 6
    def __init__(self, totalSensorBandwidth, batch_size, glimpses, optimizer, lr, momentum, discount):

        # TODO --> Integrate Discount Factor for Reward
        self.discounted_r = np.zeros((batch_size, 1))
        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.batch_size = batch_size
        self.glimpses = glimpses
        self.big_net()
        self.__build_train_fn(optimizer, lr, momentum)

      #  self.ram = self.core_network()
      #  self.gl_net = self.glimpse_network()
      #  self.act_net = self.action_network()
      #  self.loc_net = self.location_network()




    def __build_train_fn(self, opt, lr, mom):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.ram.get_layer("action_output").output
        location_prob_placeholder = self.ram.get_layer("location_output").output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        location = K.placeholder(shape=(None, self.glimpses*2),
                                                    name="sample_locations")
        baseline = K.placeholder(shape=(None, self.glimpses*2),
                                 name="baseline")

        #action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob_placeholder + 1e-10) * action_onehot_placeholder

        max_p_y = K.argmax(action_prob_placeholder)
        action = K.argmax(action_onehot_placeholder)
        R = K.equal(max_p_y, action) # reward per example
        R = K.cast(R, 'float32')
        R_out = K.reshape(R, (self.batch_size,1))

        R = K.tile(R_out, [1, (self.glimpses)*2])
        log_loc = K.log(location + 1e-10) * (R-baseline)

        loss = K.concatenate([log_action_prob, log_loc], axis=-1)
        loss = K.sum(loss, axis=-1) + K.sum(location_prob_placeholder-location_prob_placeholder, axis=-1)
        loss = loss - K.sum(K.square(R - baseline), axis=-1)
        loss = - K.mean(loss, axis=-1)


        # Choose Optimizer:
        if opt == "rmsprop":
            optimizer = keras.optimizers.rmsprop(lr=lr)
        elif opt== "adam":
            optimizer = keras.optimizers.adam(lr=lr)
        elif opt== "adadelta":
            optimizer = keras.optimizers.adadelta(lr=lr)
        elif opt== 'sgd':
            optimizer = keras.optimizers.sgd(lr=lr, momentum=mom)
        else:
            raise ValueError("Unrecognized update: {}".format(opt))

        updates = optimizer.get_updates(params=self.ram.trainable_weights,
                                   #constraints=self.ram.constraints,
                                   loss=loss)

        self.train_fn = K.function(inputs=[#self.ram.inputs,
                                           self.ram.get_layer("glimpse_input").input,
                                           self.ram.get_layer("location_input").input,
                                           action_onehot_placeholder,
                                           location,
                                           baseline
                                            ],
                                   outputs=[loss, R_out],
                                   updates=updates)


    def REINFORCE_loss(self, action_p):
        def rloss(y_true, y_pred):

            log_l = K.concatenate([K.log(action_p + 1e-10) * y_true, self.p_loc], axis=-1)
            #log_l =  K.log(self.p_loc + 1e-5) * R
            loss = K.sum(log_l, axis=-1) + K.sum(y_pred-y_pred, axis= -1)
            return - loss
        return rloss

    def merge_layer(self,inputs):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            #return K.relu(output, alpha=0.)
            return output


    def big_net(self):
        glimpse_model_i = keras.layers.Input(batch_shape=(self.batch_size, self.totalSensorBandwidth),
                                             name='glimpse_input')
        glimpse_model = keras.layers.Dense(128, activation='relu',
                                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           )(glimpse_model_i)

        glimpse_model_out = keras.layers.Dense(256,
                                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                               )(glimpse_model)

        location_model_i = keras.layers.Input(batch_shape=(self.batch_size, 2),
                                              name='location_input')
        location_model = keras.layers.Dense(128,
                                            activation = 'relu',
                                            kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            )(location_model_i)

        location_model_out = keras.layers.Dense(256,
                                            kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                )(location_model)
        #model_merge = K.relu(keras.layers.Lambda(lambda x: x[0] + x[1])([location_model_out, glimpse_model_out]))
        model_merge = keras.layers.add([location_model_out, glimpse_model_out])
        #model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        #model_merge = self.merge_layer([glimpse_model_out, location_model_out])
        glimpse_network_output  = keras.layers.Lambda(lambda x: K.relu(x))(model_merge)

      #  model_merge = keras.layers.add([glimpse_model, location_model])
      #  glimpse_network_output = keras.layers.Dense(256,
      #                                              kernel_initializer=keras.initializers.random_uniform(),
      #                                              bias_initializer=keras.initializers.random_uniform(),
      #                                              activation='relu')(model_merge)
        rnn_input = keras.layers.Reshape((256,1))(glimpse_network_output)
        model_output = keras.layers.recurrent.SimpleRNN(256,recurrent_initializer="zeros", activation='relu',
                                                return_sequences=False, stateful=True, unroll=True,
                                                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                name = 'rnn')(rnn_input)

        #model = RWA(256,recurrent_initializer="zeros", activation='relu',
        #                                         return_sequences=False, stateful=True, unroll=True,
        #                                         kernel_initializer=keras.initializers.random_uniform(),
        #                                         average_initializer = keras.initializers.random_uniform(),
        #                                         bias_initializer=keras.initializers.random_uniform(),
        #                                         name = 'rnn')(rnn_input)
        action_out = keras.layers.Dense(10,
                                 activation='softmax',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='action_output',
                                 )(model_output)
        location_out = keras.layers.Dense(2,
                                 activation='linear',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='location_output',
                                 trainable=False
                                 )(model_output)
        baseline_output = keras.layers.Dense(1,
                                 activation='sigmoid',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='baseline_output',
                                 trainable=False
                                         )(model_output)

        self.ram = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out, baseline_output])
        #self.ram.compile(optimizer=keras.optimizers.adam(lr=0.001),
       # self.ram.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False),
       #                  loss={'action_output': 'categorical_crossentropy',
       #                        'location_output': self.REINFORCE_loss(action_p=action_out)})

    def train(self, zooms, loc_input, true_a, p_loc, b):
        #b = np.stack(b)
      #  b = np.concatenate([b, b], axis=2)
      #  print b
        b = np.reshape(b, (self.batch_size, (self.glimpses) * 2))

        #self.discounted_r = np.zeros((self.batch_size, 1))
        #for b in range(self.batch_size):
        #    self.discounted_r[b] = np.sum(self.compute_discounted_R(r[b], .99))
        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))
      #  loc_input = np.reshape(loc, (self.batch_size, 2))

        loss, R = self.train_fn([glimpse_input, loc_input, true_a, p_loc, b])
        #ath = keras.utils.to_categorical(true_a, self.output_dim)
        #self.ram.fit({'glimpse_input': glimpse_input, 'location_input': loc_input},
        #                        {'action_output': ath, 'location_output': ath}, epochs=1, batch_size=self.batch_size, verbose=1, shuffle=False)
        return loss, R, np.mean(b, axis=-1)
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
        print R
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add

        discounted_r -= discounted_r.mean() / discounted_r.std()

        return discounted_r

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
    ram = RAM(totalSensorBandwidth, 32, 6)


if __name__ == '__main__':
    main()


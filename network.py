import keras
from keras import backend as K
import numpy as np

class RAM():

    def __init__(self, totalSensorBandwidth, batch_size):

        self.discounted_r = np.zeros((batch_size, 1))
        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        self.ram = self.big_net()

      #  self.ram = self.core_network()
      #  self.gl_net = self.glimpse_network()
      #  self.act_net = self.action_network()
      #  self.loc_net = self.location_network()
        self.batch_size = batch_size


#    def glimpse_network(self):
#        glimpse_model = keras.layers.Dense(128,
#                   input_dim=self.totalSensortBandwidth,
#                   activation='relu'
#            )
#        glimpse_model_out = keras.layers.Dense(256)(glimpse_model)
#
#        location_model = keras.layers.Dense(128,
#                               input_dim=2,
#                               activation = 'relu'
#                               )
#        location_model_out = keras.layers.Dense(256)(location_model)
#
#        output = keras.layers.add([glimpse_model_out, location_model_out], activation='relu')
#        return output

    def location_network(self):
        m = keras.models.Sequential()
        m.add(keras.layers.Dense(2,
                                 input_dim=256
                                 )
              )
        m.compile(optimizer=keras.optimizers.sgd(
        momentum=0.9
        ),
            loss=self.REINFORCE_loss
        )
        return m

       # model_i = keras.layers.Input(shape=(256,))
       # model = keras.layers.Dense(2)(model_i)
       # return keras.models.Model(inputs=model_i, outputs=model)



    def glimpse_network(self):
        glimpse_model_i = keras.layers.Input(shape=(self.totalSensorBandwidth,))
        glimpse_model = keras.layers.Dense(128, activation='relu')(glimpse_model_i)
        glimpse_model_out = keras.layers.Dense(256)(glimpse_model)

        location_model_i = keras.layers.Input(shape=(2,))
        location_model = keras.layers.Dense(128,
                                            activation = 'relu'
                                            )(location_model_i)
        location_model_out = keras.layers.Dense(256)(location_model)

        #model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        glimpse_network_output = keras.layers.Dense(256,
                                                activation='relu')(model_merge)
        model = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=glimpse_network_output)
        model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy')
        return model

    def core_network(self):



        model = keras.models.Sequential()

        model.add(
        keras.layers.recurrent.SimpleRNN(256, input_shape=(None,256), activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                                         recurrent_initializer='orthogonal', bias_initializer='zeros',
                                         kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                                         activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                                         bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                                         return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                                         unroll=False)
        )

        model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy')
        return model

    def action_network(self):
        m = keras.models.Sequential()
        m.add(keras.layers.Dense(10,
                                 input_dim=256,
                                 activation='softmax'
                                 )
              )
        m.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy')
        return m

    def big_net(self):
        glimpse_model_i = keras.layers.Input(shape=(None, self.totalSensorBandwidth,))
        glimpse_model = keras.layers.Dense(128, activation='relu')(glimpse_model_i)
        glimpse_model_out = keras.layers.Dense(256)(glimpse_model)

        location_model_i = keras.layers.Input(shape=(2,))
        location_model = keras.layers.Dense(128,
                                            activation = 'relu'
                                            )(location_model_i)
        location_model_out = keras.layers.Dense(256)(location_model)

        #model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        model_merge = keras.layers.add([glimpse_model_out, location_model_out])
        glimpse_network_output = keras.layers.Dense(256,
                                                    activation='relu')(model_merge)
        model = keras.layers.recurrent.SimpleRNN(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                                         recurrent_initializer='orthogonal', bias_initializer='zeros',
                                         kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                                         activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                                         bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                                         return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                                         unroll=False)(glimpse_network_output)
        action_out = keras.layers.Dense(10,
                                 activation='softmax'
                                 )(model)
        location_out = keras.layers.Dense(2,
                                 activation='linear'
                                 )(model)

        model = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out])
        return model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy', target_tensors = action_out)
        #model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy')

    def REINFORCE_loss(self, y_true, y_pred):
       # If we simply take the squared clipped diff as our loss,
       # then the gradient will be zero whenever the diff exceeds
       # the clip bounds. To avoid this, we extend the loss
       # linearly past the clip point to keep the gradient constant
       # in that regime.
       #
       # This is equivalent to declaring d loss/d q_vals to be
       # equal to the clipped diff, then backpropagating from
       # there, which is what the DeepMind implementation does.

       action_prob = y_pred* y_true
       log_action_prob = K.log(action_prob)

       assert self.discounted_r is not None, "Discounted Reward not computed"
       loss = - log_action_prob * self.discounted_r
       loss = K.mean(loss, axis=-1)
       return loss

    def train(self, zooms, loc, glimpse, ram_s, act_prob, true_a, r):
        self.discounted_r = np.zeros((self.batch_size, 1))
        for b in range(self.batch_size):
            self.discounted_r[b] = np.sum(self.compute_discounted_R(r[b], .99))


        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))
        action = np.argmax(act_prob, axis=-1)
        ah = self.dense_to_one_hot(action)
        ath = self.dense_to_one_hot(true_a)
        self.gl_net.train_on_batch([glimpse_input, loc], ath)
        self.ram.train_on_batch(glimpse, ath)
        self.act_net.train_on_batch(ram_s, ath)
        self.loc_net.train_on_batch(act_prob, ath)

    def __build_train_fn_REINFORCE(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.act_net.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None, ),
                                                    name="discount_reward")

        #TODO: Summation is wrong?
        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        opt = keras.optimizers.sgd(momentum=0.)

        updates = opt.get_updates(params=self.ram.trainable_weights,
                                   constraints=[],
                                   loss='mse')

        self.train_fn = keras.function(inputs=[self.ram.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

      #  updates_loc_net = opt.get_updates(params=self.loc_net.trainable_weights,
      #                             constraints=[],
      #                             loss='mse')

      #  self.train_fn_loc_net = keras.function(inputs=[self.ram.input,
      #                                         action_onehot_placeholder,
      #                                         discount_reward_placeholder],
      #                                 outputs=[],
      #                                 updates=updates_loc_net)

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

        discounted_r -= discounted_r.mean() / discounted_r.std()

        return discounted_r

    def fit_loc_net(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = keras.utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = self.compute_discounted_R(R)

        assert S.shape[1] == self.totalSensorBandwidth, "{} != {}".format(S.shape[1], self.totalSensorBandwidth)
        assert action_onehot.shape[0] == S.shape[0], "{} != {}".format(action_onehot.shape[0], S.shape[0])
        assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
        assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))

        self.train_fn_loc_net([S, action_onehot, discount_reward])

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
        gl_out = self.gl_net.predict_on_batch([glimpse_input, loc])
        ram_out = self.ram.predict_on_batch(np.reshape(gl_out, (self.batch_size, 1, 256)))
       # print self.act_net.predict_on_batch(ram_out)
       # print self.loc_net.predict_on_batch(ram_out)

        return self.act_net.predict_on_batch(ram_out), self.loc_net.predict_on_batch(ram_out), ram_out, gl_out


    def get_best_action(self,prob_a):
        return np.argmax(prob_a)

def main():
    totalSensorBandwidth = 3 * 8 * 8 * 1
    ram = RAM(totalSensorBandwidth, 32)


if __name__ == '__main__':
    main()


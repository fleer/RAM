import keras
from keras import backend as K
import numpy as np

class RAM():

    def __init__(self, totalSensorBandwidth, batch_size):

        self.output_dim = 10
        self.totalSensorBandwidth = totalSensorBandwidth
        initial_loc = np.random.uniform(-1, 1,(batch_size, 2))
        self.ram = self.core_network()
        self.gl_net = self.glimpse_network()
        self.act_net = self.action_network()
        self.loc_net = self.location_network()
        self.__build_train_fn_REINFORCE_action()


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
        model_i = keras.layers.Input(shape=(256,))
        model = keras.layers.Dense(2)(model_i)
        return keras.models.Model(inputs=model_i, outputs=model)


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
        model_merge = keras.layers.merge([glimpse_model_out, location_model_out], mode='sum')
        glimpse_network_output = keras.layers.Dense(256,
                                                activation='relu')(model_merge)
        train_model = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=glimpse_network_output)
        train_model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False), loss='categorical_crossentropy')

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

        model_i = keras.layers.Input(shape=(256,))
        model = keras.layers.Dense(10, activation='softmax')(model_i)
        return keras.models.Model(inputs=model_i, outputs=model)



    def __build_train_fn_REINFORCE_action(self):
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

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = keras.optimizers.adam()

      #  updates = adam.get_updates(params=self.ram.trainable_weights,
      #                             constraints=[],
      #                             loss=loss)

      #  self.train_fn = keras.function(inputs=[self.ram.input,
      #                                     action_onehot_placeholder,
      #                                     discount_reward_placeholder],
      #                             outputs=[],
      #                             updates=updates)

        updates_loc_net = adam.get_updates(params=self.loc_net.trainable_weights,
                                   constraints=[],
                                   loss=loss)

        self.train_fn_loc_net = keras.function(inputs=[self.ram.input,
                                               action_onehot_placeholder,
                                               discount_reward_placeholder],
                                       outputs=[],
                                       updates=updates_loc_net)

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

def main():
    totalSensorBandwidth = 3 * 8 * 8 * 1
    ram = RAM(totalSensorBandwidth, 32)


if __name__ == '__main__':
    main()


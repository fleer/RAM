import keras
from keras import backend as K
import numpy as np
import os

class RAM():
    """
    Neural Network class, that uses KERAS to build and trains the Recurrent Attention Model
    """


    def __init__(self, totalSensorBandwidth, batch_size, glimpses, lr, lr_decay, min_lr, loc_std):
        """
        Intialize parameters, determine the learning rate decay and build the RAM
        :param totalSensorBandwidth: The length of the networks input vector
                                    ---> nZooms * sensorResolution * sensorResolution * channels
        :param batch_size: Size of each batch
        :param glimpses: Number of glimpses the model executes on each image
        :param optimizer: The used optimize: "sgd, rmsprop, adadelta, adam,..."
        :param lr: The learning rate at epoch e=0
        :param lr_decay: Number of epochs after which the learning rate has linearly
                        decayed to min_lr
        :param min_lr: minimal learning rate
        :param momentum: should momentum be used
        :param loc_std: standard deviation of location policy
        :param clipnorm: Gradient clipping
        :param clipvalue: Gradient clipping
        """

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

        # Create the network
       # self.big_net(optimizer, lr, momentum, clipnorm, clipvalue)


    def big_net(self, optimizer, lr, momentum, clipnorm, clipvalue):
        """
        Function to create the Recurrent Attention Model and compile it for the different
        Loss Functions
        :param optimizer: The used optimize: "sgd, rmsprop, adadelta, adam,..."
        :param lr: The learning rate at epoch e=0
        :param momentum: should momentum be used
        :param clipnorm: Gradient clipping
        :param clipvalue: Gradient clipping
        :return: None
        """

        #   ================
        #   Glimpse Network
        #   ================

        # Build the glimpse input
        glimpse_model_i = keras.layers.Input(batch_shape=(self.batch_size, self.totalSensorBandwidth),
                                             name='glimpse_input')
        glimpse_model = keras.layers.Dense(128, activation='relu',
                                           kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                           name='glimpse_1'
                                           )(glimpse_model_i)

      #  glimpse_model_out = keras.layers.Dense(256,
      #                                     kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
      #                                     bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
      #                                         name='glimpse_3'
      #                                         )(glimpse_model)

        # Build the location input
        location_model_i = keras.layers.Input(batch_shape=(self.batch_size, 2),
                                              name='location_input')

        location_model = keras.layers.Dense(128,
                                            activation = 'relu',
                                            kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                            name='location_1'
                                            )(location_model_i)

      #  location_model_out = keras.layers.Dense(256,
      #                                      kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
      #                                      bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
      #                                          name='location_2'
      #                                          )(location_model)

      #  model_merge = keras.layers.add([glimpse_model_out, location_model_out], name='add')
      #  glimpse_network_output  = keras.layers.Lambda(lambda x: K.relu(x))(model_merge)
        model_concat = keras.layers.concatenate([location_model, glimpse_model])
        glimpse_network_output_0  = keras.layers.Dense(256,
                                                      activation = 'relu',
                                                      kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                      bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
                                                      )(model_concat)
        glimpse_network_output  = keras.layers.Dense(256,
                                                     activation = 'linear',
                                                     kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                     bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
                                                     )(glimpse_network_output_0)
        #   ================
        #   Core Network
        #   ================
        rnn_input = keras.layers.Reshape((256,1))(glimpse_network_output)
        model_output = keras.layers.recurrent.SimpleRNN(256,recurrent_initializer="zeros", activation='relu',
                                                return_sequences=False, stateful=True, unroll=True,
                                                kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                                name = 'rnn')(rnn_input)
        #   ================
        #   Action Network
        #   ================
        action_out = keras.layers.Dense(10,
                                 activation='softmax',
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='action_output',
                                 )(model_output)
        #   ================
        #   Location Network
        #   ================

        location_out = keras.layers.Dense(2,
                                 activation='tanh',
                                 #kernel_initializer=keras.initializers.glorot_uniform(),
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                # bias_initializer=keras.initializers.glorot_uniform(),
                                 name='location_output',
                                 )(model_output)
        #   ================
        #   Baseline Network
        #   ================
        baseline_output = keras.layers.Dense(1,
                                 activation='sigmoid',
                               #  kernel_initializer=keras.initializers.glorot_uniform(),
                                 kernel_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 bias_initializer=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
                                 name='baseline_output',
                                         )(model_output)

        # Create the model
        self.ram = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out, baseline_output])

        # Compile the model
        if optimizer == "rmsprop":
            self.ram.compile(optimizer=keras.optimizers.rmsprop(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                             loss={'action_output': self.CROSS_ENTROPY,
                                   'location_output': self.REINFORCE_LOSS(action_p=action_out, baseline=baseline_output),
                                   'baseline_output': self.BASELINE_LOSS(action_p=action_out)})
        elif optimizer == "adam":
            self.ram.compile(optimizer=keras.optimizers.adam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                             loss={'action_output': self.CROSS_ENTROPY,
                                   'location_output': self.REINFORCE_LOSS(action_p=action_out, baseline=baseline_output),
                                   'baseline_output': self.BASELINE_LOSS(action_p=action_out)})
        elif optimizer == "adadelta":
            self.ram.compile(optimizer=keras.optimizers.adadelta(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                             loss={'action_output': self.CROSS_ENTROPY,
                                   'location_output': self.REINFORCE_LOSS(action_p=action_out, baseline=baseline_output),
                                   'baseline_output': self.BASELINE_LOSS(action_p=action_out)})
        elif optimizer == 'sgd':
            self.ram.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=False, clipvalue=clipvalue, clipnorm=clipnorm),
                             loss={'action_output': self.CROSS_ENTROPY,
                                   'location_output': self.REINFORCE_LOSS(action_p=action_out, baseline=baseline_output),
                                   'baseline_output': self.BASELINE_LOSS(action_p=action_out)})
        else:
            raise ValueError("Unrecognized update: {}".format(optimizer))

        # Print Summary
        self.ram.summary()

    def CROSS_ENTROPY(self, y_true, y_pred):
        """
        Standard CrossEntropy Loss
        :param y_true: True Value
        :param y_pred: Network Prediction
        :return: Loss
        """
      #  self.ram.trainable = True
        return K.categorical_crossentropy(y_true, y_pred)

    def REINFORCE_LOSS(self, action_p, baseline):
        """
        :param action_p: Network output of action network
        :param baseline: Network putput of baseline network
        :return: Loss, based on REINFORCE algorithm for the normal
                distribution
        """
        def loss(y_true, y_pred):
            """
            REINFORCE algorithm for Normal Distribution
            Used for location network loss
            -------
            Williams, Ronald J. "Simple statistical gradient-following
            algorithms for connectionist reinforcement learning."
            Machine learning 8.3-4 (1992): 229-256.
            -------

            Here, some tricks are used to get the desired result...
            :param y_true:  One-Hot Encoding of correct Action
            :param y_pred: Output of Location Network --> Mean of the Normal distribution
            :return: Loss, based on REINFORCE algorithm for the normal
                     distribution
            """
            # Compute Predicted and Correct action values
            max_p_y = K.argmax(action_p)
            action = K.argmax(y_true)

            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R = K.cast(R, 'float32')
            R_out = K.reshape(R, (self.batch_size,1))

            #Uses the REINFORCE algorithm in sec 6. p.237-239)
            # Individual loss for location network
            # Compute loss via REINFORCE algorithm
            # for gaussian distribution
            # d ln(f(m,s,x))   (x - m)
            # -------------- = -------- with m = mean, x = sample, s = standard_deviation
            #       d m          s**2

            #Sample Location, based on current mean
            sample_loc = K.random_normal(y_pred.shape, y_pred, self.loc_std)

            #TODO: Check how to deal with the 2 dims (x,y) of location
            R = K.tile(R_out, [1, 2])
            b = K.tile(baseline, [1, 2])
            loss_loc = ((sample_loc - y_pred)/(self.loc_std*self.loc_std)) * (R -b)
            return - loss_loc
      #  self.ram.trainable = False
      #  self.ram.get_layer('location_output').trainable = True
        return loss

    def BASELINE_LOSS(self, action_p):
        """
        :param action_p: Network output of action network
        :return: Baseline Loss
        """
        def loss(y_true, y_pred):
            """
            The baseline is trained with mean-squared-error
            The only difficulty is to use the current reward
            as the true value

            :param y_true:  One-Hot Encoding of correct Action
            :param y_pred:  Output of Baseline Network
            :return: Baseline Loss
            """
            # Compute Predicted and Correct action values
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
        """
        Function to control the linear decay
        of the learning rate
        :return: New learning rate
        """
        lr = K.get_value(self.ram.optimizer.lr)
        # Linear Learning Rate Decay
        lr = max(self.min_lr, lr - self.lr_decay_rate)
        K.set_value(self.ram.optimizer.lr, lr)
        return lr

    def train(self, zooms, loc_input, true_a):
        """
        Train the Model!
        :param zooms: Current zooms, created using loc_input
        :param loc_input: Current Location
        :param true_a: One-Hot Encoding of correct action
        :return: Average Loss of training step
        """
      #  self.ram.trainable = True


        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))

        loss = self.ram.train_on_batch({'glimpse_input': glimpse_input, 'location_input': loc_input},
                                       {'action_output': true_a, 'location_output': true_a, 'baseline_output': true_a})

        #return loss, loss_l, loss_b, R#, np.mean(b, axis=-1)
        return np.mean(loss)

    def reset_states(self):
        """
        Reset the hidden state of the Core Network
        :return:
        """
        self.ram.reset_states()

    def choose_action(self,X,loc):
        """
        Choose action and new location, based on current
        network state
        :param X: Current Batch
        :param loc: New Location
        :return: Output of Action Network & Location Network
        """

        glimpse_input = np.reshape(X, (self.batch_size, self.totalSensorBandwidth))
        action_prob, loc, _ = self.ram.predict_on_batch({"glimpse_input": glimpse_input, 'location_input': loc})
        #return self.act_net.predict_on_batch(ram_out), self.loc_net.predict_on_batch(ram_out), ram_out, gl_out
        return action_prob, loc

    def save_model(self, path, filename):
        """
        Saves the model to ``model.json`` file and
        also the model weights to model.h5 file

        :param path: Path to file
        :param filename: Filename
        :return:
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

    def load_model(self, path, filename, optimizer, lr, momentum, clipnorm, clipvalue):
        """
        Load the model from ``model.json`` file and
        also the model weights from model.h5 file

        :param path: Path to file
        :param filename: Filename
        :return: Loading successfull
        """
        model_fn = os.path.join(path, filename)
        if os.path.isfile(model_fn + '.json') and os.path.isfile(model_fn + '.h5'):
            # load json and create model
            json_file = open(model_fn + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.ram = keras.models.model_from_json(loaded_model_json)
            # load weights into new model
            self.ram.load_weights(model_fn + ".h5")

            # Compile the model
            if optimizer == "rmsprop":
                self.ram.compile(optimizer=keras.optimizers.rmsprop(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                                 loss={'action_output': self.CROSS_ENTROPY,
                                       'location_output': self.REINFORCE_LOSS(action_p=self.ram.get_layer("action_output").output, baseline=self.ram.get_layer("baseline_output").output),
                                       'baseline_output': self.BASELINE_LOSS(action_p=self.ram.get_layer("action_output").output)})
            elif optimizer == "adam":
                self.ram.compile(optimizer=keras.optimizers.adam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                                 loss={'action_output': self.CROSS_ENTROPY,
                                       'location_output': self.REINFORCE_LOSS(action_p=self.ram.get_layer("action_output").output, baseline=self.ram.get_layer("baseline_output").output),
                                       'baseline_output': self.BASELINE_LOSS(action_p=self.ram.get_layer("action_output").output)})
            elif optimizer == "adadelta":
                self.ram.compile(optimizer=keras.optimizers.adadelta(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm),
                                 loss={'action_output': self.CROSS_ENTROPY,
                                       'location_output': self.REINFORCE_LOSS(action_p=self.ram.get_layer("action_output").output, baseline=self.ram.get_layer("baseline_output").output),
                                       'baseline_output': self.BASELINE_LOSS(action_p=self.ram.get_layer("action_output").output)})
            elif optimizer == 'sgd':
                self.ram.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=False, clipvalue=clipvalue, clipnorm=clipnorm),
                                 loss={'action_output': self.CROSS_ENTROPY,
                                       'location_output': self.REINFORCE_LOSS(action_p=self.ram.get_layer("action_output").output, baseline=self.ram.get_layer("baseline_output").output),
                                       'baseline_output': self.BASELINE_LOSS(action_p=self.ram.get_layer("action_output").output)})
            else:
                raise ValueError("Unrecognized update: {}".format(optimizer))

            self.ram.summary()
            return True
        else:
            return False


def main():
    """
    Test the written Code
    :return:
    """
    totalSensorBandwidth = 3 * 8 * 8 * 1
    ram = RAM(totalSensorBandwidth, 32, 6, "sdg", 0.001, 20, 0.0001, 0.9, 0.11, 1, 1)
    ram.save_model("./", "test")
    print "Model saved..."
    if ram.load_model("./", "test"):
        print "Model loaded..."


if __name__ == '__main__':
    main()


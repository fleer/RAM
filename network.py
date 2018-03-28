import os

import keras
import numpy as np
from keras import backend as K


class RAM():
    """
    Neural Network class, that uses KERAS to build and trains the Recurrent Attention Model
    """


    def __init__(self, totalSensorBandwidth, batch_size, glimpses, pixel_scaling, lr, lr_decay, min_lr, loc_std):
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
        self.pixel_scaling = pixel_scaling
        self.average_loc = K.variable(np.zeros((self.batch_size, 2)))
        self.average_baseline = K.variable(np.zeros((self.batch_size, 1)))
        # Learning Rate Decay
        if self.lr_decay != 0:
            self.lr_decay_rate = ((lr - min_lr) /
                                 lr_decay)


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

        init_kernel = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        bias_initializer = keras.initializers.Zeros()

        # Build the glimpse input
        glimpse_model_i = keras.layers.Input(batch_shape=(self.batch_size, self.totalSensorBandwidth),
                                             name='glimpse_input')
        glimpse_model = keras.layers.Dense(128, activation='relu',
                                           kernel_initializer=init_kernel,
                                           bias_initializer=bias_initializer,
                                           name='glimpse_1'
                                           )(glimpse_model_i)

        # Build the location input
        location_model_i = keras.layers.Input(batch_shape=(self.batch_size, 2),
                                              name='location_input')

        location_model = keras.layers.Dense(128,
                                            activation = 'relu',
                                            kernel_initializer=init_kernel,
                                            bias_initializer=bias_initializer,
                                            name='location_1'
                                            )(location_model_i)

        model_concat = keras.layers.concatenate([location_model, glimpse_model])

        glimpse_network_output_0  = keras.layers.Dense(256,
                                                      activation = 'relu',
                                                      kernel_initializer=init_kernel,
                                                      bias_initializer=bias_initializer,
                                                      )(model_concat)
        glimpse_network_output  = keras.layers.Dense(256,
                                                     activation = 'linear',
                                                     kernel_initializer=init_kernel,
                                                     bias_initializer=bias_initializer,
                                                     )(glimpse_network_output_0)
        #   ================
        #   Core Network
        #   ================
        rnn_input = keras.layers.Reshape((256,1))(glimpse_network_output)
        model_output = keras.layers.SimpleRNN(256,recurrent_initializer="zeros", activation='relu',
       # model_output = keras.layers.LSTM(256, recurrent_initializer="zeros", activation='relu',
                                                return_sequences=False, stateful=True, unroll=True,
                                                kernel_initializer=init_kernel,
                                                bias_initializer=bias_initializer,
                                                name = 'rnn')(rnn_input)
        #   ================
        #   Action Network
        #   ================
        action_out = keras.layers.Dense(10,
                                 activation=self.log_softmax,
                                 kernel_initializer=init_kernel,
                                 bias_initializer=bias_initializer,
                                 name='action_output',
                                 )(model_output)

        stop_grad = keras.layers.Lambda(lambda x: K.stop_gradient(x))(model_output)
        #   ================
        #   Location Network
        #   ================

        location_out = keras.layers.Dense(2,
                                 activation=self.hard_tanh,
                                 kernel_initializer=init_kernel,
                                 bias_initializer=bias_initializer,
                                 name='location_output'
                                 )(stop_grad)

        #   ================
        #   Baseline Network
        #   ================
        baseline_output = keras.layers.Dense(1,
                                 activation='sigmoid',
                                 kernel_initializer=init_kernel,
                                 bias_initializer=bias_initializer,
                                 name='baseline_output',
                                         )(stop_grad)

        # Create the model
        self.ram = keras.models.Model(inputs=[glimpse_model_i, location_model_i], outputs=[action_out, location_out, baseline_output])

        # Print Summary
        self.ram.summary()

        #   ================
        #   Location Network at timestep 0
        #   ================

        #TODO: Find a better solution

        hidden_state_in_0 = keras.layers.Input(shape=(256,))
        location_out_t0 = keras.layers.Dense(2,
                                             activation=self.hard_tanh,
                                             kernel_initializer=init_kernel,
                                             bias_initializer=bias_initializer,
                                             name='location_output_t0',
                                             trainable=False
                                             )(hidden_state_in_0)

        # Create Location model at timestep 0
        self.loc_t0 = keras.models.Model(inputs=hidden_state_in_0, outputs=location_out_t0)

        # Compile the model
        if optimizer == "rmsprop":
            opt = keras.optimizers.rmsprop(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm)
        elif optimizer == "adam":
            opt = keras.optimizers.adam(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm)
        elif optimizer == "adadelta":
            opt = keras.optimizers.adadelta(lr=lr, clipvalue=clipvalue, clipnorm=clipnorm)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True, clipvalue=clipvalue, clipnorm=clipnorm)
        else:
            raise ValueError("Unrecognized update: {}".format(optimizer))
        self.ram.compile(optimizer=opt,
                         loss={'action_output': self.total_loss_a(baseline=baseline_output, mean=location_out),
                               'location_output': self.total_loss_l(action_p=action_out, baseline=baseline_output),
                                'baseline_output': self.baseline_loss(action_p=action_out)})

        # Print Summary
        self.ram.summary()

    def hard_tanh(self, x):
        """Segment-wise linear approximation of tanh.

         Faster than tanh.
         Returns `-1.` if `x < -1.`, `1.` if `x > 1`.
         In `-1. <= x <= 1.`, returns `x`.

         # Arguments
             x: A tensor or variable.

         # Returns
             A tensor.
         """
        import tensorflow as tf

        lower = tf.convert_to_tensor(-1., x.dtype.base_dtype)
        upper = tf.convert_to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, lower, upper)
        return x

    def log_softmax(self, x, axis=-1):
        import tensorflow as tf
        return tf.nn.log_softmax(x)
        #TODO: Check own log_softmax implementation
        #return x - K.log(K.sum(K.exp(x), axis=axis, keepdims=True))

    def total_loss_l(self, action_p, baseline):
        """
        :param action_p: Network output of action network
        :param baseline: Network putput of baseline network
        :return: Loss, based on REINFORCE algorithm for the normal
                distribution
        """
#        self.ram.trainable = True
#        self.ram.get_layer('location_output').trainable = True
        action_p = K.stop_gradient(action_p)
        baseline = K.stop_gradient(baseline)
        baseline = K.mean(K.concatenate([self.average_baseline, baseline], axis=1), axis=1)
        baseline = K.reshape(baseline, (self.batch_size,1))
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
            max_p_y = K.argmax(action_p, axis=-1)
            #max_p_y = K.stack([max_p_y, max_p_y], axis=-1)
            max_p_y = K.reshape(max_p_y,(self.batch_size,1))
            max_p_y = K.tile(max_p_y,[1,2])
            #action = K.argmax(y_true, axis=-1)
            action = K.cast(y_true, 'int64')

            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R = K.cast(R, 'float32')
            #R_out = K.reshape(R, (self.batch_size,1))
            policy = action_p * K.one_hot(action[:,0], self.output_dim)

            #Uses the REINFORCE algorithm in sec 6. p.237-239)
            # Individual loss for location network
            # Compute loss via REINFORCE algorithm
            # for gaussian distribution
            # d ln(f(m,s,x))   (x - m)
            # -------------- = -------- with m = mean, x = sample, s = standard_deviation
            #       d m          s**2
         #   g = y_pred + K.random_normal(shape=K.shape(y_pred), mean=0., stddev=self.loc_std)
         #   location = self.hard_tanh(g) * self.pixel_scaling # normLoc coordinates are between -1 and 1

            #TODO: Check how to deal with the 2 dims (x,y) of location
            #R = K.tile(R_out, [1, 2])
            #R = K.stack([R_out, R_out], axis=-1 )
            b = K.tile(baseline, [1, 2])
            # b = K.stack([baseline, baseline], axis=-1 )
            #loss_loc = ((self.average_loc - y_pred)/(self.loc_std*self.loc_std)) * (R -b)
            loss_loc = self.average_loc * (R -b)
            cost = K.concatenate([policy, loss_loc], axis=1)

            cost = K.sum(cost, axis=1) + K.sum(y_pred - y_pred)
            cost = K.mean(cost, axis=0)
            return - cost
        #TODO: Test alternative--> Only train dense layer of location output
        #self.ram.trainable = False
        #self.ram.get_layer('location_mean').trainable = True
        return loss

    def total_loss_a(self, baseline, mean):
        """
        :param action_p: Network output of action network
        :param baseline: Network putput of baseline network
        :return: Loss, based on REINFORCE algorithm for the normal
                distribution
        """
 #       self.ram.trainable = True
        baseline = K.stop_gradient(baseline)
        baseline = K.mean(K.concatenate([self.average_baseline, baseline], axis=1), axis=1)
        baseline = K.reshape(baseline, (self.batch_size,1))
        mean = K.stop_gradient(mean)
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
            max_p_y = K.argmax(y_pred, axis=-1)
            action = K.argmax(y_true, axis=-1)

            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R_out = K.cast(R, 'float32')
           # R_out = K.reshape(R, (self.batch_size,1))

            #Uses the REINFORCE algorithm in sec 6. p.237-239)
            # Individual loss for location network
            # Compute loss via REINFORCE algorithm
            # for gaussian distribution
            # d ln(f(m,s,x))   (x - m)
            # -------------- = -------- with m = mean, x = sample, s = standard_deviation
            #       d m          s**2
          #  g = mean + K.random_normal(shape=K.shape(mean), mean=0., stddev=self.loc_std)
          #  location = self.hard_tanh(g) * self.pixel_scaling # normLoc coordinates are between -1 and 1

            #TODO: Check how to deal with the 2 dims (x,y) of location
           # R = K.tile(R_out, [1, 2])
            R = K.stack([R_out, R_out], axis=-1 )
            b = K.tile(baseline, [1, 2])
           # b = K.stack([baseline, baseline], axis=-1 )
            #loss_loc = ((self.average_loc - mean)/(self.loc_std*self.loc_std)) * (R -b)
            loss_loc = self.average_loc * (R -b)
            cost = K.concatenate([y_pred * y_true, loss_loc], axis=1)
            cost = K.sum(cost, axis=1)
            cost = K.mean(cost, axis=0)
            return - cost
        #TODO: Test alternative--> Only train dense layer of location output
        #self.ram.trainable = False
        #self.ram.get_layer('location_mean').trainable = True
        return loss

    def baseline_loss(self, action_p):
        """
        :param action_p: Network output of action network
        :return: Baseline Loss
        """
       # self.ram.trainable = False
       #  self.ram.get_layer('baseline_output').trainable = True
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
            max_p_y = K.argmax(action_p, axis=-1)
            action = K.cast(y_true, 'int64')
          #  baseline = np.mean(K.concatenate([self.average_baseline, y_pred], axis=-1), axis=-1)
          #  baseline = K.reshape(baseline, (self.batch_size,1))
            # Get Reward for current step
            R = K.equal(max_p_y, action) # reward per example
            R_out = K.cast(R, 'float32')
            return K.mean(K.square(y_pred - R_out), axis=-1)
        return loss

    def set_av_loc(self,av_loc):
        K.set_value(self.average_loc, av_loc)

    def set_av_b(self,av_b):
        K.set_value(self.average_baseline, av_b)

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

    def train(self, zooms, loc_input, Y):
        """
        Train the Model!
        :param zooms: Current zooms, created using loc_input
        :param loc_input: Current Location
        :param true_a: One-Hot Encoding of correct action
        :return: Average Loss of training step
        """
        #self.ram.trainable = True

        # A little bit hacky, but we need the reward in the loss function
        # instead of the location
        loc_reward = np.stack([Y,Y],axis=-1)

        one_hot = keras.utils.to_categorical(Y, 10)
        glimpse_input = np.reshape(zooms, (self.batch_size, self.totalSensorBandwidth))

     #   w = self.ram.get_layer("location_output").get_weights()
        loss = self.ram.train_on_batch({'glimpse_input': glimpse_input, 'location_input': loc_input},
                                       {'action_output': one_hot, 'location_output': loc_reward,
                                        'baseline_output': np.reshape(Y, (self.batch_size,1))})
     #   w1 = self.ram.get_layer("location_output").get_weights()
     #   for blub in range(len(w)):
     #       if abs(w[0][0][blub] - w1[0][0][blub]) > 1e-6:
     #           print("change")


        return np.mean(loss)

    def reset_states(self):
        """
        Reset the hidden state of the Core Network
        :return:
        """
        self.ram.reset_states()
    def start_location(self):
        w = self.ram.get_layer("location_output").get_weights()
        self.loc_t0.set_weights(w)
        h0 = np.zeros((self.batch_size,256))
        return self.loc_t0.predict_on_batch(h0)


    def choose_action(self,X,loc):
        """
        Choose action and new location, based on current
        network state
        :param X: Current Batch
        :param loc: New Location
        :return: Output of Action Network & Location Network
        """

        glimpse_input = np.reshape(X, (self.batch_size, self.totalSensorBandwidth))
        action_prob, loc, b = self.ram.predict_on_batch({"glimpse_input": glimpse_input, 'location_input': loc})
        return action_prob, loc, b

    def get_weights(self):
        return self.ram.get_weights()

    def set_weights(self, weights):
        return self.ram.set_weights(weights)

    def save_model(self, path, filename):
        """
        Saves the model weights to model.h5 file

        :param path: Path to file
        :param filename: Filename
        :return:
        """
        model_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        # serialize weights to HDF5
        self.ram.save_weights(model_fn)

    def load_model(self, path, filename):
        """
        Load the model weights from model.h5 file

        :param path: Path to file
        :param filename: Filename
        :return: Loading successfull
        """
        model_fn = os.path.join(path, filename)
        if  os.path.isfile(model_fn):
            # load weights into new model
            self.ram.load_weights(model_fn)
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


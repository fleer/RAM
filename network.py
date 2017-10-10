import keras
import numpy as np

class RAM():

    def __init__(self, totalSensorBandwidth):
        self.totalSensortBandwidth = totalSensorBandwidth
        pass

    def glimpse_network(self):
        glimpse_model = keras.layers.Dense(128,
                   input_dim=self.totalSensortBandwidth,
                   activation='relu'
            )
        glimpse_model_out = keras.layers.Dense(256)(glimpse_model)

        location_model = keras.layers.Dense(128,
                               input_dim=2,
                               activation = 'relu'
                               )
        location_model_out = keras.layers.Dense(256)(location_model)

        output = keras.layers.add([glimpse_model_out, location_model_out], activation='relu')
        model = keras.models.Model([glimpse_model, location_model], outputs=output)
        return model

    def core_network(self, batch_size):
        initial_loc = np.random.uniform(-1, 1,(batch_size, 2))
        keras.layers.recurrent.SimpleRNN(256, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform',
                                         recurrent_initializer='orthogonal', bias_initializer='zeros',
                                         kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                                         activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                                         bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                                         return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                                         unroll=False)



def main():
    totalSensorBandwidth = 3 * 8 * 8 * 1

if __name__ == '__main__':
    main()


from MNIST_Processing import MNIST
from network import RAM
import numpy as np
import keras
from collections import defaultdict
import logging
import time
import os
import json

class Experiment():
    """
    Main class, controlling the experiment
    """


    #   ================
    #   Evaluation
    #   ================

    results = defaultdict(list)


    def __init__(self, PARAMETERS, DOMAIN_OPTIONS, results_file="001-results.json", model_file="001-network"):

        logging.basicConfig(level=logging.INFO)

        mnist_size = DOMAIN_OPTIONS.MNIST_SIZE
        channels = DOMAIN_OPTIONS.CHANNELS
        minRadius = DOMAIN_OPTIONS.MIN_ZOOM
        sensorResolution = DOMAIN_OPTIONS.SENSOR
        self.loc_std = DOMAIN_OPTIONS.LOC_STD
        self.nZooms = DOMAIN_OPTIONS.NZOOMS
        self.nGlimpses = DOMAIN_OPTIONS.NGLIMPSES

        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_epochs = PARAMETERS.MAX_EPOCHS

        totalSensorBandwidth = self.nZooms * sensorResolution * sensorResolution * channels
        self.mnist = MNIST(mnist_size, self.batch_size, channels, minRadius, sensorResolution,
                           self.nZooms, self.loc_std, DOMAIN_OPTIONS.UNIT_PIXELS,
                           DOMAIN_OPTIONS.TRANSLATE, DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
        self.ram = RAM(totalSensorBandwidth, self.batch_size, self.nGlimpses,
                       PARAMETERS.OPTIMIZER, PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
                       PARAMETERS.MIN_LEARNING_RATE, PARAMETERS.MOMENTUM,
                       DOMAIN_OPTIONS.LOC_STD, PARAMETERS.CLIPNORM, PARAMETERS.CLIPVALUE)

        if PARAMETERS.LOAD_MODEL:
            if self.ram.load_model(PARAMETERS.MODEL_FILE_PATH, PARAMETERS.MODEL_FILE):
                logging.info("Loaded model from " + PARAMETERS.MODEL_FILE_PATH + PARAMETERS.MODEL_FILE)

        self.train(PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY)
        self.save('./', results_file)
        if model_file is not None:
            self.ram.save_model('./', model_file)

    def performance_run(self, total_epochs):
        actions = 0.
        actions_sqrt = 0.
        num_data = len(self.mnist.dataset.test._images)
        batches_in_epoch = num_data // self.batch_size

        for i in xrange(batches_in_epoch):
            X, Y= self.mnist.get_batch_test(self.batch_size)
            loc = np.random.uniform(-1, 1,(self.batch_size, 2))
            sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            for n in range(self.nGlimpses):
                zooms = self.mnist.glimpseSensor(X,sample_loc)
                a_prob, loc = self.ram.choose_action(zooms, sample_loc)
                sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            action = np.argmax(a_prob, axis=-1)
            actions += np.sum(np.equal(action,Y).astype(np.float32))
            actions_sqrt += np.sum((np.equal(action,Y).astype(np.float32))**2, axis=-1)
            self.ram.reset_states()

        accuracy = actions/num_data
        accuracy_std = np.sqrt(((actions_sqrt/num_data) - accuracy**2)/num_data)

        self.results['learning_steps'].append(total_epochs)
        self.results['accuracy'].append(accuracy)
        self.results['accuracy_std'].append(accuracy_std)

        return accuracy, accuracy_std

    def train(self, lr, lr_decay):
        total_epochs = 0
        data = self.mnist.dataset.train
        batches_in_epoch = len(data._images) // self.batch_size

        # Initial Performance Check
        accuracy, accuracy_std = self.performance_run(total_epochs)
        logging.info("Epoch={:d}: >>> Accuracy: {:.4f} "
                     "+/- {:.6f}".format(total_epochs, accuracy, accuracy_std))


        for i in range(self.max_epochs):
            start_time = time.time()
            accuracy = 0
            while total_epochs == self.mnist.dataset.train.epochs_completed:
                X, Y= self.mnist.get_batch_train(self.batch_size)
                loc = np.random.uniform(-1, 1, (self.batch_size, 2))
                sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
                for n in range(1, self.nGlimpses):
                    zooms = self.mnist.glimpseSensor(X, sample_loc)
                    a_prob, loc = self.ram.choose_action(zooms, sample_loc)
                    sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
                zooms = self.mnist.glimpseSensor(X, sample_loc)
                ath = keras.utils.to_categorical(Y, 10)
                loss = self.ram.train(zooms, sample_loc, ath)
                action = np.argmax(a_prob, axis=-1)
                accuracy += np.mean(np.equal(action,Y).astype(np.float32))
                self.ram.reset_states()
            total_epochs += 1
            if lr_decay > 0:
                lr = self.ram.learning_rate_decay()

            # Check Performance
           # if total_steps % (self.max_epochs / self.num_policy_checks) == 0:

            accuracy, accuracy_std = self.performance_run(total_epochs)

           # elif self.mnist.dataset.train.epochs_completed % 1 == 0:
            logging.info("Epoch={:d}: >>> training time/epoch: {:.2f}, Loss: {:.4f}, "
                         "Learning Rate: {:.6f}, Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                                 (time.time()-start_time), loss,
                                                 lr, accuracy, accuracy_std))

    def save(self, path, filename):
        """
        Saves the experimental results to ``results.json`` file
        """
        results_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(results_fn, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=True)
        f.close()

    def __del__(self):
        self.results.clear()

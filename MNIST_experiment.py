from MNIST_Processing import MNIST
from network import RAM
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import logging
import time
import os
import sys
import json

class Experiment():
    """
    Main class, controlling the experiment
    """

    results = defaultdict(list)


    def __init__(self, PARAMETERS, DOMAIN_OPTIONS, results_file="results.json", model_file="network.h5"):

        logging.basicConfig(level=logging.INFO)

        #   ================
        #   Reading the parameters
        #   ================

        mnist_size = DOMAIN_OPTIONS.MNIST_SIZE
        channels = DOMAIN_OPTIONS.CHANNELS
        scaling_factor = DOMAIN_OPTIONS.SCALING_FACTOR
        sensorResolution = DOMAIN_OPTIONS.SENSOR
        self.loc_std = DOMAIN_OPTIONS.LOC_STD
        self.nZooms = DOMAIN_OPTIONS.DEPTH
        self.nGlimpses = DOMAIN_OPTIONS.NGLIMPSES

        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_epochs = PARAMETERS.MAX_EPOCHS

        totalSensorBandwidth = self.nZooms * sensorResolution * sensorResolution * channels

        #   ================
        #   Loading the MNIST Dataset
        #   ================

        self.mnist = MNIST(mnist_size, self.batch_size, channels, scaling_factor, sensorResolution,
                           self.nZooms, self.loc_std, DOMAIN_OPTIONS.UNIT_PIXELS,
                           DOMAIN_OPTIONS.TRANSLATE, DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)

        #   ================
        #   Creating the RAM
        #   ================

        if DOMAIN_OPTIONS.TRANSLATE:
            self.pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)
        else:
            self.pixel_scaling = (DOMAIN_OPTIONS.UNIT_PIXELS * 2.)/ float(DOMAIN_OPTIONS.MNIST_SIZE)

        self.ram = RAM(totalSensorBandwidth, self.batch_size, self.nGlimpses, self.pixel_scaling,
                       PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
                       PARAMETERS.MIN_LEARNING_RATE, DOMAIN_OPTIONS.LOC_STD)

        self.ram.big_net(PARAMETERS.OPTIMIZER,PARAMETERS.LEARNING_RATE,PARAMETERS.MOMENTUM,
                         PARAMETERS.CLIPNORM, PARAMETERS.CLIPVALUE)

        if PARAMETERS.LOAD_MODEL:
            if self.ram.load_model(PARAMETERS.MODEL_FILE_PATH, PARAMETERS.MODEL_FILE):
                logging.info("Loaded model weigths from " + PARAMETERS.MODEL_FILE_PATH + PARAMETERS.MODEL_FILE + "!")
            else:
                logging.info("Weigts from " + PARAMETERS.MODEL_FILE_PATH + PARAMETERS.MODEL_FILE + " could not be loaded!")
                sys.exit(0)


        #   ================
        #   Train
        #   ================

        self.train(PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY, PARAMETERS.EARLY_STOPPING, PARAMETERS.PATIENCE)
        self.save('./', results_file)
        if model_file is not None:
            self.ram.save_model('./', model_file)

    def performance_run(self, total_epochs, validation=False):
        """
        Function for evaluating the current model on the
        validation- or test-dataset

        :param total_epochs: Number of trained epochs
        :param validation: Should the smaller validation-dataset
                be evaluated
        :return: current accuracy and its error
        """
        actions = 0.
        actions_sqrt = 0.
        if validation:
            num_data = len(self.mnist.dataset.validation._images)
            batches_in_epoch = num_data // self.batch_size
        else:
            num_data = len(self.mnist.dataset.test._images)
            batches_in_epoch = num_data // self.batch_size

        for i in xrange(batches_in_epoch):
            if validation:
                X, Y= self.mnist.get_batch_validation(self.batch_size)
            else:
                X, Y= self.mnist.get_batch_test(self.batch_size)
            mean_loc = self.ram.start_location()
            loc = np.maximum(-1., np.minimum(1., np.random.normal(mean_loc, self.loc_std, mean_loc.shape)))* self.pixel_scaling
            for n in range(self.nGlimpses):
                zooms = self.mnist.glimpseSensor(X,loc)
                a_prob, mean_loc, _ = self.ram.choose_action(zooms, loc)
                #loc = mean_loc * self.pixel_scaling
                # During evaluation, instead of sampling from the normal distribution, the output is
                # taken to be the input, i.e. the mean.
                loc = np.maximum(-1., np.minimum(1., np.random.normal(mean_loc, self.loc_std, mean_loc.shape)))* self.pixel_scaling
                #sample_loc = np.maximum(-1., np.minimum(1., loc))
            action = np.argmax(a_prob, axis=-1)
            actions += np.sum(np.equal(action,Y).astype(np.float32), axis=-1)
            actions_sqrt += np.sum((np.equal(action,Y).astype(np.float32))**2, axis=-1)
            self.ram.reset_states()

        accuracy = actions/num_data
        accuracy_std = np.sqrt(((actions_sqrt/num_data) - accuracy**2)/num_data)

        if not validation:
            # Save to results file
            self.results['learning_steps'].append(total_epochs)
            self.results['accuracy'].append(accuracy)
            self.results['accuracy_std'].append(accuracy_std)

        return accuracy, accuracy_std

    def train(self, lr, lr_decay, early_stopping, patience):
        """
        Training the current model
        :param lr: learning rate
        :param lr_decay: Number of steps the Learning rate should (linearly)
        :param early_stopping: Use early stopping
        :param patience: Number of Epochs observing the worsening of
                Validation set, before stopping
        :return:
        """
        total_epochs = 0
        # Initial Performance Check
        accuracy, accuracy_std = self.performance_run(total_epochs)
        logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} "
                      "+/- {:.6f}".format(total_epochs, accuracy, accuracy_std))
        num_train_data = len(self.mnist.dataset.train._images)

        patience_steps = 0
        early_stopping_accuracy = 0.
        best_weights = None

        for i in range(self.max_epochs):
            mean_loss = []
            start_time = time.time()
            test_accuracy = 0
            test_accuracy_sqrt = 0
            while total_epochs == self.mnist.dataset.train.epochs_completed:
                average_loc = []
                average_b = []
                X, Y= self.mnist.get_batch_train(self.batch_size)
                loc = self.ram.start_location()
                sample_loc = np.maximum(-1., np.minimum(1., np.random.normal(loc, self.loc_std, loc.shape))) * self.pixel_scaling
                loc_pdf = stats.norm(loc, self.loc_std).logpdf(sample_loc)
                for n in range(1, self.nGlimpses):
                    average_loc.append(loc_pdf)
                    zooms = self.mnist.glimpseSensor(X, sample_loc)
                    a_prob, loc, b = self.ram.choose_action(zooms, sample_loc)
                    # During training, the output is sampled from a normal distribution with fixed standard deviation.
                    sample_loc = np.maximum(-1., np.minimum(1., np.random.normal(loc, self.loc_std, loc.shape)))*self.pixel_scaling
                    loc_pdf = stats.norm(loc, self.loc_std).logpdf(sample_loc)
                    average_b.append(b)
                average_loc.append(loc_pdf)
                self.ram.set_av_loc(np.mean(average_loc, axis=0))
                self.ram.set_av_b(np.mean(average_b, axis=0))
                zooms = self.mnist.glimpseSensor(X, sample_loc)
                loss = self.ram.train(zooms, sample_loc, Y)
                mean_loss.append(loss)
                action = np.argmax(a_prob, axis=-1)
                test_accuracy += np.sum(np.equal(action,Y).astype(np.float32), axis=-1)
                test_accuracy_sqrt+= np.sum((np.equal(action,Y).astype(np.float32))**2, axis=-1)
                self.ram.reset_states()
            total_epochs += 1
            if lr_decay > 0:
                lr = self.ram.learning_rate_decay()

            # Check Performance
            if total_epochs % 10 == 0:
                # Test Accuracy
                accuracy, accuracy_std = self.performance_run(total_epochs)

                # Print out Infos
                logging.info("Epoch={:d}: >>> Test-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs, accuracy, accuracy_std))
            else:
                # Validation Accuracy
                accuracy, accuracy_std = self.performance_run(total_epochs, validation=True)

                # Test Accuracy
                test_accuracy = test_accuracy/num_train_data
                test_accuracy_std = np.sqrt(((test_accuracy_sqrt/num_train_data) - test_accuracy**2)/num_train_data)

                # Print out Infos
                logging.info("Epoch={:d}: >>> examples/s: {:.2f}, Loss: {:.4f}, "
                             "Learning Rate: {:.6f}, Train-Accuracy: {:.4f} +/- {:.6f}, "
                             "Validation-Accuracy: {:.4f} +/- {:.6f}".format(total_epochs,
                                 float(num_train_data)/float(time.time()-start_time), np.mean(mean_loss),
                                 lr, test_accuracy, test_accuracy_std, accuracy, accuracy_std))

                # Early Stopping
                if early_stopping and early_stopping_accuracy < accuracy:
                    early_stopping_accuracy = accuracy
                    best_weights = self.ram.get_weights()
                    patience_steps = 0
                else:
                    patience_steps += 1

            if patience_steps > patience:
                self.ram.set_weights(best_weights)
                logging.info("Early Stopping at Epoch={:d}! Validation Accuracy is not increasing. The best Newtork will be saved!".format(total_epochs))
                return 0

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

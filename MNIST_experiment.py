from MNIST_Domain import MNIST
from network import RAM
import numpy as np
import keras
from collections import defaultdict
import logging
import time
import os
import json

import matplotlib.pyplot as plt

class Experiment():
    """
    Main class, controlling the experiment
    """


    #   ================
    #   Evaluation
    #   ================

    results = defaultdict(list)


    def __init__(self, PARAMETERS, DOMAIN_OPTIONS, results_file="./results0.json"):

        logging.basicConfig(level=logging.INFO)

        mnist_size = DOMAIN_OPTIONS.MNIST_SIZE
        channels = DOMAIN_OPTIONS.CHANNELS
        minRadius = DOMAIN_OPTIONS.MIN_ZOOM
        sensorResolution = DOMAIN_OPTIONS.SENSOR
        self.loc_std = DOMAIN_OPTIONS.LOC_STD
        self.nZooms = DOMAIN_OPTIONS.NZOOMS
        self.nGlimpses = DOMAIN_OPTIONS.NGLIMPSES

        self.batch_size = PARAMETERS.BATCH_SIZE
        self.max_steps = PARAMETERS.MAX_STEPS
        self.num_policy_checks = PARAMETERS.NUM_POLICY_CHECKS

        totalSensorBandwidth = self.nZooms * sensorResolution * sensorResolution * channels
        self.mnist = MNIST(mnist_size, self.batch_size, channels, minRadius, sensorResolution, self.nZooms, self.loc_std)
        self.ram = RAM(totalSensorBandwidth, self.batch_size, self.nGlimpses,
                       PARAMETERS.OPTIMIZER, PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
                       PARAMETERS.MOMENTUM, PARAMETERS.DISCOUNT, DOMAIN_OPTIONS.LOC_STD, PARAMETERS.CLIPNORM, PARAMETERS.CLIPVALUE)
        # What happens to learning rate
        self.lr = PARAMETERS.LEARNING_RATE
        self.lr_decay = PARAMETERS.LEARNING_RATE_DECAY

        self.train()
        self.save('./', results_file)

    def performance_run(self, total_steps):
        actions = 0.
        data = self.mnist.dataset.test
        batches_in_epoch = len(data._images) // self.batch_size

        for i in xrange(batches_in_epoch):
            X, Y = self.mnist.dataset.test.next_batch(self.batch_size)
            loc = np.random.uniform(-1, 1,(self.batch_size, 2))
            sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            for n in range(self.nGlimpses):
                zooms = self.mnist.glimpseSensor(X,sample_loc)
                a_prob, loc = self.ram.choose_action(zooms, sample_loc)
                sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            action = np.argmax(a_prob, axis=-1)
            actions += np.mean(np.equal(action,Y).astype(np.float32))
            self.ram.reset_states()

        self.results['learning_steps'].append(total_steps)
        self.results['return'].append(actions/float(batches_in_epoch))

        logging.info("Total Steps={:d}: >>> Accuracy: {:.4f}".format(total_steps, actions/float(batches_in_epoch)))

    def train(self):
        total_steps = 0

        # Initial Performance Check
  #      self.performance_run(total_steps)

        for i in range(self.max_steps):
            start_time = time.time()

            X, Y= self.mnist.get_batch(self.batch_size)
            baseline = np.zeros((self.batch_size, self.nGlimpses, 2))
            loc = np.random.uniform(-1, 1, (self.batch_size, 2))
            sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            for n in range(1, self.nGlimpses):
                zooms = self.mnist.glimpseSensor(X, sample_loc)
                a_prob, loc = self.ram.choose_action(zooms, sample_loc)
                sample_loc = np.tanh(np.random.normal(loc, self.loc_std, loc.shape))
            zooms = self.mnist.glimpseSensor(X, sample_loc)
            ath = keras.utils.to_categorical(Y, 10)
            loss_a, loss_l, loss_b, R = self.ram.train(zooms, sample_loc, ath)
            self.ram.reset_states()
           # if total_steps % 20 == 0:
           #     print "Action_L: {}, Location_L: {}, Baseline_L: {}".format(loss_a, loss_l, loss_b)
            total_steps += 1

            # Check Performance
            if total_steps % (self.max_steps / self.num_policy_checks) == 0:

                self.performance_run(total_steps)

            if total_steps % 100 == 0:
                lr = self.lr
                if self.lr_decay > 0:
                    lr *= 1. / (1. + self.lr_decay * total_steps)
                #logging.info("Total Steps={:d}: >>> steps/second: {:.2f}, average loss: {:.4f}, "
                #             "Reward: {:.2f}".format(total_steps,
                #             1./(time.time()-start_time), loss_l, np.mean(R)))
                logging.info("Total Steps={:d}: >>> steps/second: {:.2f}, Action_L: {:.4f}, Location_L: {:.4f}, Baseline_L: {:.4f}, "
                             "Learning Rate: {:.6f}, Reward: {:.2f}".format(total_steps,
                                                     1./(time.time()-start_time),loss_a, loss_l, loss_b, lr, np.mean(R)))

    def save(self, path, filename):
        """Saves the experimental results to ``results.json`` file
        """
        results_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(results_fn, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=True)
        f.close()

def main():
    for i in range(11):
        exp = Experiment("./results" + "{0:03}".format(i) + ".json")
        exp = None

if __name__ == '__main__':
    main()
#img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
#fig = plt.figure()
#plt.ion()
#plt.show()
#
#initial_loc = np.random.uniform(-1, 1,(batch_size, 2))
#
#zooms = mnist.glimpseSensor(X, initial_loc)
#
#for k in xrange(batch_size):
#    one_img = img[k,:,:,:]
#    max_radius = minRadius * (2 ** (depth - 1))
#    offset = max_radius
#    one_img = mnist.pad_to_bounding_box(one_img, offset, offset,
#                                        max_radius * 2 + mnist_size, max_radius * 2 + mnist_size)
#
#    plt.title(Y[k], fontsize=40)
#    plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
#               interpolation="nearest")
#
#    plt.draw()
#    #time.sleep(0.05)
#    plt.pause(1.0001)
#
#    for z in zooms[k]:
#        plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
#                   interpolation="nearest")
#
#        plt.draw()
#        #time.sleep(0.05)
#        plt.pause(1.0001)

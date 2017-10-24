from MNIST_Domain import MNIST
from network import RAM
import numpy as np
import keras
from collections import defaultdict
import os
import json

import matplotlib.pyplot as plt

class Experiment():


    def __init__(self):

        mnist_size = 28
        batch_size = 32
        channels = 1 # grayscale
        minRadius = 4 # zooms -> minRadius * 2**<depth_level>
        sensorResolution = 8 # fixed resolution of sensor
        nZooms = 3 # zooms
        totalSensorBandwidth = nZooms * sensorResolution * sensorResolution * channels

        nGlimpse = 6
        mnist = MNIST(mnist_size, batch_size, channels, minRadius, sensorResolution, nZooms)
        ram = RAM(totalSensorBandwidth, batch_size, nGlimpse)

        loc_sd = 0.11               # std when setting the location

        results = defaultdict(list)

        epoch = 0
        num_policy_checks = 20
        max_steps = 100000



    def performance_run(self):
        actions = []
        data = mnist.dataset.test
        batches_in_epoch = len(data._images) // batch_size
        accuracy = 0

        for i in xrange(batches_in_epoch):
            X, Y = mnist.dataset.test.next_batch(batch_size)
            loc = np.random.uniform(-1, 1,(batch_size, 2))
            sample_loc = np.tanh(loc + np.random.normal(0, loc_sd, loc.shape))
            mean_locs = np.zeros((batch_size, nGlimpse, 2))
            sample_locs = np.zeros((batch_size, nGlimpse, 2))
            for i in range(batch_size):
                mean_locs[i][0][0] = loc[i][0]
                mean_locs[i][0][1] = loc[i][1]
                sample_locs[i][0][0] = sample_loc[i][0]
                sample_locs[i][0][1] = sample_loc[i][1]
            for n in range(nGlimpse):
                zooms = mnist.glimpseSensor(X,sample_loc)
                a_prob, loc, _ = ram.choose_action(zooms, sample_loc)
                mean_loc = np.fmax(-1.0, np.fmin(1.0, loc + sample_locs[:,-1]))
                sample_loc = np.fmax(-1.0, np.fmin(1.0, mean_loc + np.random.normal(0, loc_sd, loc.shape)))
                for i in range(batch_size):
                    mean_locs[i][n][0] = mean_loc[i][0]
                    mean_locs[i][n][1] = mean_loc[i][1]
                    sample_locs[i][n][0] = sample_loc[i][0]
                    sample_locs[i][n][1] = sample_loc[i][1]
            action = np.argmax(a_prob, axis=-1)
            actions += np.mean(np.equal(action,Y).astype(np.float32))
            ram.reset_states()

        results['learning_steps'].append(epoch)
        results['return'].append(actions/float(batches_in_epoch))

        print "Accuracy: {}".format(np.mean(actions))

for i in range(100000):
    # Check Performance
    if total_steps % (self.max_steps / self.num_policy_checks) == 0:

        self.performance_run(total_steps, episode_number)

    if epoch % 5000 == 0:
    X, Y= mnist.get_batch(batch_size)
    baseline = np.zeros((batch_size, nGlimpse, 2))
    mean_locs = np.zeros((batch_size, nGlimpse, 2))
    sample_locs = np.zeros((batch_size, nGlimpse, 2))
    loc = np.random.uniform(-1, 1, (batch_size, 2))
    sample_loc = np.tanh(loc + np.random.normal(0, loc_sd, loc.shape))
    for i in range(batch_size):
        mean_locs[i][0][0] = loc[i][0]
        mean_locs[i][0][1] = loc[i][1]
        sample_locs[i][0][0] = sample_loc[i][0]
        sample_locs[i][0][1] = sample_loc[i][1]

    for n in range(1, nGlimpse):
        zooms = mnist.glimpseSensor(X, sample_loc)
        a_prob, loc, bl = ram.choose_action(zooms, sample_loc)
        mean_loc = np.fmax(-1.0, np.fmin(1.0, loc + sample_locs[:,-1]))
        sample_loc = np.fmax(-1.0, np.fmin(1.0, mean_loc + np.random.normal(0, loc_sd, loc.shape)))
        for i in range(batch_size):
                mean_locs[i][n][0] = mean_loc[i][0]
                mean_locs[i][n][1] = mean_loc[i][1]
                sample_locs[i][n][0] = sample_loc[i][0]
                sample_locs[i][n][1] = sample_loc[i][1]
                baseline[i][n] = bl[i]

    zooms = mnist.glimpseSensor(X, sample_loc)
   # max_p_y = np.argmax(a_prob, axis=-1)
   # R = np.equal(max_p_y, Y.astype('int64')) # reward per example
   # R = np.reshape(R, (batch_size, 1))
   # R = R.astype('float32')
    p_loc = ram.gaussian_pdf(mean_locs.reshape((batch_size, nGlimpse * 2)), sample_locs.reshape((batch_size, nGlimpse * 2)))
    p_loc = np.tanh(p_loc)
    #p_loc = np.reshape(p_loc, (batch_size, nGlimpse * 2))
    ath = keras.utils.to_categorical(Y, 10)

   # l2= np.log(p_loc) * R
   # l1 = np.log(a_prob) * ath
   # c = np.concatenate([l1,l2], axis=-1)
   # d = np.sum(c, axis=-1)
    loss, R, b = ram.train(zooms, sample_loc, ath, p_loc, baseline)
    ram.reset_states()

    epoch += 1
    if epoch % 200 == 0:
        #print "Epoch: {} --> Correct guess: {} --> Baseline: {} --> Loss: {}".format(epoch, np.mean(np.equal(np.argmax(a_prob, axis=-1),Y).astype(np.float32)), b, loss)
        print "Epoch: {} --> Reward: {} --> R-B: {}" \
              " --> Loss: {}".format(epoch, np.mean(R), np.mean(R)-np.mean(b), loss)

"""Saves the experimental results to ``results.json`` file
"""
results_fn = os.path.join('./', 'results.json')
with open(results_fn, "w") as f:
    json.dump(results, f, indent=4, sort_keys=True)
f.close()

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

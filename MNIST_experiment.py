from MNIST_Domain import MNIST
from network import RAM
import numpy as np

import matplotlib.pyplot as plt

mnist_size = 28
batch_size = 10
channels = 1 # grayscale
minRadius = 4 # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8 # fixed resolution of sensor
depth = 3 # zooms
totalSensorBandwidth = depth * sensorBandwidth * sensorBandwidth * channels

nGlimps = 6
mnist = MNIST(mnist_size,batch_size,channels,minRadius,sensorBandwidth,depth)
ram = RAM(totalSensorBandwidth, batch_size, nGlimps)

loc_sd = 0.11               # std when setting the location


epoch = 0
for i in range(500000):
    if epoch % 5000 == 1:
        actions = []
        data = mnist.dataset.test
        batches_in_epoch = len(data._images) // batch_size
        accuracy = 0

        for i in xrange(batches_in_epoch):
            X, Y = mnist.dataset.test.next_batch(batch_size)
            loc = np.random.uniform(-1, 1,(batch_size, 2))

            for n in range(nGlimps):

                sample_loc = np.fmax(-1.0, np.fmin(1.0, loc + np.random.normal(0, loc_sd, loc.shape)))
                zooms = mnist.glimpseSensor(X,sample_loc)
                a_prob, loc = ram.choose_action(zooms, sample_loc)
                    #
            action = np.argmax(a_prob, axis=-1)
            actions.append(np.equal(action,Y).astype(np.float32))
            ram.reset_states()
        print "Accuracy: {}".format(np.mean(actions))
    ram.mean_locs = []
    ram.sampled_locs = []
    X, Y= mnist.get_batch(batch_size)
    loc = np.random.uniform(-1, 1,(batch_size, 2))

    for n in range(nGlimps-1):
        ram.mean_locs.append(loc)
        sample_loc = np.maximum(-1.0, np.minimum(1.0, loc + np.random.normal(0, loc_sd, loc.shape)))
        ram.sampled_locs.append(sample_loc)
        zooms = mnist.glimpseSensor(X, sample_loc)
        a_prob, loc = ram.choose_action(zooms, sample_loc)

    #
    #    r = np.equal(action,Y).astype(np.float32)
    #    for j in range(batch_size):
    #        R[j].append(r[j])
    ram.mean_locs.append(loc)
    action = np.argmax(a_prob, axis=-1)
    sample_loc = np.maximum(-1.0, np.minimum(1.0, loc + np.random.normal(0, loc_sd, loc.shape)))
    ram.sampled_locs.append(sample_loc)
    zooms = mnist.glimpseSensor(X, sample_loc)
    ram.train(zooms, sample_loc, Y)
    ram.reset_states()

    epoch += 1
    if epoch % 25 == 0:
        print "Epoch: {} --> Correct guess: {}".format(epoch, np.mean(np.equal(action,Y).astype(np.float32)))


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

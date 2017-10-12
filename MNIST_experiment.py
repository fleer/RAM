from MNIST_Domain import MNIST
from network import RAM
import numpy as np

import matplotlib.pyplot as plt

mnist_size = 28
batch_size = 2
channels = 1 # grayscale
minRadius = 4 # zooms -> minRadius * 2**<depth_level>
sensorBandwidth = 8 # fixed resolution of sensor
depth = 8 # zooms
totalSensorBandwidth = depth * sensorBandwidth * sensorBandwidth * channels

nGlimps = 6
mnist = MNIST(mnist_size,batch_size,channels,minRadius,sensorBandwidth,depth)
ram = RAM(totalSensorBandwidth, batch_size)
X, Y= mnist.get_batch(batch_size)

mean_locs = []
loc_sd = 0.11               # std when setting the location

initial_loc = np.random.uniform(-1, 1,(batch_size, 2))
mean_locs.append(initial_loc)
R = [[0 for x in range(1)] for y in range(batch_size)]

ram.ram.reset_states()
zooms =  mnist.glimpseSensor(X,initial_loc)
a_prob, loc, _, _ = ram.choose_action(zooms, initial_loc)
action = np.argmax(a_prob, axis=-1)
#Compute reward
r = np.equal(action,Y).astype(np.float32)
for j in range(batch_size):
    R[j].append(r[j])
    del R[j][0]
for n in range(nGlimps-1):
    sample_loc = np.maximum(-1.0, np.minimum(1.0, loc + np.random.normal(0, loc_sd, loc.shape)))
    zooms = mnist.glimpseSensor(X,sample_loc)
    a_prob, loc, ram_out, gl_out = ram.choose_action(zooms, sample_loc)

    mean_locs.append(loc)
    action = np.argmax(a_prob, axis=-1)
    r = np.equal(action,Y).astype(np.float32)
    for j in range(batch_size):
        R[j].append(r[j])
    #R.append(np.cast(np.equal(action,Y), np.float32))
ram.train(zooms, loc, gl_out, ram_out, a_prob, Y, R)


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

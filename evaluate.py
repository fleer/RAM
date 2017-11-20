import sys
from network import RAM
from MNIST_Processing import MNIST
from matplotlib import pyplot as plt
import numpy as np

# This is not a nice way to implement the different configuration scripts...
if len(sys.argv) > 2:
    if sys.argv[1] == 'run_mnist':
        from run_mnist import MNIST_DOMAIN_OPTIONS
        from run_mnist import PARAMETERS
    elif sys.argv[1] == 'run_translated_mnist':
        from run_translated_mnist import MNIST_DOMAIN_OPTIONS
        from run_translated_mnist import PARAMETERS
    else:
        print "Wrong file name for confiuration file!"
        sys.exit(0)
else:
    print "Give Configuration File as additional argument! \n " \
          "E.g. python evaluate.py run_mnist ./model/001-network"
    sys.exit(0)


def glimpseSensor(self, img, normLoc):
    assert not np.any(np.isnan(normLoc))," Locations have to be between 1, -1: {}".format(normLoc)
    assert np.any(np.abs(normLoc)<=1)," Locations have to be between 1, -1: {}".format(normLoc)

    #loc = np.around(((normLoc + 1) / 2.0) * self.mnist_size) # normLoc coordinates are between -1 and 1
    loc = normLoc * (self.unit_pixels * 2.)/ self.mnist_size # normLoc coordinates are between -1 and 1
    # Convert location [-1,1] into MNIST Coordinates:
    loc = np.around(((loc + 1) / 2.0) * self.mnist_size) # normLoc coordinates are between -1 and 1
    #print "Default: {}".format(loc)
    #print "unit Pixels: {}".format(loc1)
    loc = loc.astype(np.int32)

    img = np.reshape(img, (self.batch_size, self.mnist_size, self.mnist_size, self.channels))

    zooms = []

    # process each image individually
    for k in xrange(self.batch_size):
        imgZooms = []
        one_img = img[k,:,:,:]
        max_radius = self.mnist_size * (self.minRadius ** (self.depth - 1))
        offset = 2 * max_radius

        # pad image with zeros
        one_img = self.pad_to_bounding_box(one_img, offset, offset, \
                                           max_radius * 4 + self.mnist_size, max_radius * 4 + self.mnist_size)

        for i in xrange(self.depth):
            r = int(self.mnist_size * (self.minRadius ** (i - 1)))

            d_raw = 2 * r
            d = np.reshape(np.asarray(d_raw), [1])

            d = np.tile(d, [2])

            loc_k = loc[k,:]
            adjusted_loc = offset + loc_k - r


            one_img2 = np.reshape(one_img, (one_img.shape[0], \
                                            one_img.shape[1]))

            # crop image to (d x d)
            #zoom = slice(one_img2, adjusted_loc, d)
            zoom = one_img2[adjusted_loc[0]:adjusted_loc[0]+d[0], adjusted_loc[1]:adjusted_loc[1]+d[1]]
            assert not np.any(np.equal(zoom.shape, (0,0))), "Picture has size 0, location {}, depth {}".format(adjusted_loc, d)

            imgZooms.append(zoom)

        zooms.append(np.stack(imgZooms))

    zooms = np.stack(zooms)

    return zooms


save = True

mnist_size = MNIST_DOMAIN_OPTIONS.MNIST_SIZE
channels = MNIST_DOMAIN_OPTIONS.CHANNELS
minRadius = MNIST_DOMAIN_OPTIONS.MIN_ZOOM
sensorResolution = MNIST_DOMAIN_OPTIONS.SENSOR
loc_std = MNIST_DOMAIN_OPTIONS.LOC_STD
nZooms = MNIST_DOMAIN_OPTIONS.NZOOMS
nGlimpses = MNIST_DOMAIN_OPTIONS.NGLIMPSES

#Reduce the batch size for evaluatoin
batch_size = PARAMETERS.BATCH_SIZE

totalSensorBandwidth = nZooms * sensorResolution * sensorResolution * channels
mnist = MNIST(mnist_size, batch_size, channels, minRadius, sensorResolution,
                   nZooms, loc_std, MNIST_DOMAIN_OPTIONS.UNIT_PIXELS,
                   MNIST_DOMAIN_OPTIONS.TRANSLATE, MNIST_DOMAIN_OPTIONS.TRANSLATED_MNIST_SIZE)

ram = RAM(totalSensorBandwidth, batch_size, nGlimpses,
               PARAMETERS.LEARNING_RATE, PARAMETERS.LEARNING_RATE_DECAY,
               PARAMETERS.MIN_LEARNING_RATE, MNIST_DOMAIN_OPTIONS.LOC_STD)

if ram.load_model('./', sys.argv[2], PARAMETERS.OPTIMIZER,PARAMETERS.LEARNING_RATE,PARAMETERS.MOMENTUM,
                       PARAMETERS.CLIPNORM, PARAMETERS.CLIPVALUE):
    print("Loaded model from " + sys.argv[2] +
                 ".json and " + sys.argv[2] + ".h5!")
else:
    print("Model from " + sys.argv[2] +
                 " could not be loaded!")
    sys.exit(0)

plt.ion()
plt.show()

X, Y= mnist.get_batch_test(batch_size)
img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
for k in xrange(batch_size):
    one_img = img[k,:,:,:]

    plt.title(Y[k], fontsize=40)
    plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
               interpolation="nearest")
    plt.draw()
    #time.sleep(0.05)
    if save:
        plt.savefig("symbol_" + repr(k) + ".png")
    plt.pause(1.0001)

loc = np.random.uniform(-1, 1,(batch_size, 2))
sample_loc = np.tanh(np.random.normal(loc, loc_std, loc.shape))
for n in range(nGlimpses):
    zooms = mnist.glimpseSensor(X,sample_loc)
    ng = 1
    #for g in zooms:
    for g in range(batch_size):
        nz = 1
        plt.title(Y[g], fontsize=40)
        for z in zooms[g]:
        #for z in g:
            plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
                       interpolation="nearest")

            plt.draw()
            if save:
                plt.savefig("symbol_" + repr(g) + "_" +
                            "glimpse_" + repr(n) + "_" +
                            "zoom_" + repr(nz) + ".png")
            #time.sleep(0.05)
            plt.pause(1.0001)
            nz += 1
        ng += 1
    a_prob, loc = ram.choose_action(zooms, sample_loc)
    sample_loc = np.tanh(np.random.normal(loc, loc_std, loc.shape))
ram.reset_states()



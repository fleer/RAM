import tf_mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MNIST():

    def __init__(self, mnist_size, batch_size, channels, minRadius, sensorBandwidth,depth, loc_std, unit_pixels):

        self.mnist_size = mnist_size
        self.batch_size = batch_size
        self.channels = channels # grayscale
        self.minRadius = minRadius # zooms -> minRadius * 2**<depth_level>
        self.sensorBandwidth = sensorBandwidth # fixed resolution of sensor
        self.sensorArea = self.sensorBandwidth**2
        self.depth = depth # zooms
        self.unit_pixels = unit_pixels
        self.dataset = tf_mnist_loader.read_data_sets("mnist_data")

        self.loc_std = loc_std # std when setting the location

    def get_batch(self, batch_size):
        X, Y = self.dataset.test.next_batch(batch_size)
        return X,Y

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


                one_img2 = np.reshape(one_img, (one_img.shape[0],\
                    one_img.shape[1]))

                # crop image to (d x d)
                #zoom = slice(one_img2, adjusted_loc, d)
                zoom = one_img2[adjusted_loc[0]:adjusted_loc[0]+d[0], adjusted_loc[1]:adjusted_loc[1]+d[1]]
                assert not np.any(np.equal(zoom.shape, (0,0))), "Picture has size 0, location {}, depth {}".format(adjusted_loc, d)
                #zoom = np.reshape(zoom, (1, d_raw, d_raw, 1))


                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = cv2.resize(zoom, (self.sensorBandwidth, self.sensorBandwidth),
                      interpolation=cv2.INTER_AREA)#INTER_LINEAR)
                zoom = np.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                imgZooms.append(zoom)

            zooms.append(np.stack(imgZooms))

        zooms = np.stack(zooms)

        return zooms

    def pad_to_bounding_box(self, image, offset_height, offset_width, target_height,
                            target_width):
        """Pad `image` with zeros to the specified `height` and `width`.
        Adds `offset_height` rows of zeros on top, `offset_width` columns of
        zeros on the left, and then pads the image on the bottom and right
        with zeros until it has dimensions `target_height`, `target_width`.
        This op does nothing if `offset_*` is zero and the image already has size
        `target_height` by `target_width`.
        Args:
          image: 4-D Tensor of shape `[batch, height, width, channels]` or
                 3-D Tensor of shape `[height, width, channels]`.
          offset_height: Number of rows of zeros to add on top.
          offset_width: Number of columns of zeros to add on the left.
          target_height: Height of output image.
          target_width: Width of output image.
        Returns:
          If `image` was 4-D, a 4-D float Tensor of shape
          `[batch, target_height, target_width, channels]`
          If `image` was 3-D, a 3-D float Tensor of shape
          `[target_height, target_width, channels]`
        Raises:
          ValueError: If the shape of `image` is incompatible with the `offset_*` or
            `target_*` arguments, or either `offset_height` or `offset_width` is
            negative.
        """

        is_batch = True
        image_shape = image.shape
        if image.ndim == 3:
            is_batch = False
            image = np.expand_dims(image, 0)
        elif image.ndim is None:
            is_batch = False
            image = np.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image_shape.ndims != 4:
            raise ValueError('\'image\' must have either 3 or 4 dimensions.')

        batch = len(image)
        height = len(image[0])
        width = len(image[0,0])
        depth = len(image[0,0,0])

        after_padding_width = target_width - offset_width - width
        after_padding_height = target_height - offset_height - height

        assert offset_height >= 0, 'offset_height must be >= 0'
        assert offset_width >= 0, 'offset_width must be >= 0'
        assert after_padding_width >= 0, 'width must be <= target - offset'
        assert after_padding_height >= 0, 'height must be <= target - offset'

        # Do not pad on the depth dimensions.
        paddings = np.reshape(
               np.stack([
                0, 0, offset_height, after_padding_height, offset_width,
                after_padding_width, 0, 0
            ]), [4, 2])
        padded = np.pad(image, paddings, 'constant', constant_values=0)

        padded_shape = [i for i in [batch, target_height, target_width, depth]]
        np.reshape(padded, padded_shape)

        if not is_batch:
            padded = np.squeeze(padded, axis=0)

        return padded



def main():
    mnist = MNIST(28,4,1,2,6,1,0.11, 13)
    mnist_size = mnist.mnist_size
    batch_size = mnist.batch_size
    channels = 1 # grayscale

    minRadius = 4 # zooms -> minRadius * 2**<depth_level>
    sensorBandwidth = 8 # fixed resolution of sensor
    sensorArea = sensorBandwidth**2
    depth = 1 # zooms

    glimpses = 4
    X, Y= mnist.get_batch(batch_size)
    #mnist.glimpseSensor(X, )


    img = np.reshape(X, (batch_size, mnist_size, mnist_size, channels))
    fig = plt.figure()
    plt.ion()
    plt.show()

    #initial_loc = np.random.uniform(-1, 1,(batch_size, 2))

#    zooms = mnist.glimpseSensor(X, initial_loc)
    zooms = [mnist.glimpseSensor(X, np.random.uniform(-1, 1,(batch_size, 2))) for x in range(glimpses)]

    for k in xrange(batch_size):
        one_img = img[k,:,:,:]
        max_radius = mnist_size * (minRadius ** (depth - 1))
        offset = max_radius

        one_img = mnist.pad_to_bounding_box(one_img, offset, offset,
                                            max_radius * 2 + mnist_size, max_radius * 2 + mnist_size)

        plt.title(Y[k], fontsize=40)
        plt.imshow(one_img[:,:,0], cmap=plt.get_cmap('gray'),
                   interpolation="nearest")

        plt.draw()
        #time.sleep(0.05)
        plt.pause(1.0001)

        for g in zooms:
            #for z in zooms[k]:
            for z in g[k]:
                plt.imshow(z[:,:], cmap=plt.get_cmap('gray'),
                       interpolation="nearest")

                plt.draw()
                #time.sleep(0.05)
                plt.pause(1.0001)

if __name__ == '__main__':
    main()

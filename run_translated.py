from MNIST_experiment import Experiment
import numpy as np

class MNIST_DOMAIN_OPTIONS:
    """
    Class for the Setup of the Domain Parameters
    """
    # Size of each image: MNIST_SIZE x MNIST_SIZE
    MNIST_SIZE = 28
    #
    #   ================
    #   Reward constants
    #   ================
    #   Reward for correctly Identifying a number:
    REWARD = +1.
    #   Step Reward

    #   ======================
    #   Domin specific options
    #   ======================
    #
    # Number of image channels: 1
    # --> greyscale
    CHANNELS = 1
    #
    # Resolution of the Sensor
    SENSOR = 12
    # Number of zooms
    NZOOMS = 3
    # zoom sale # zooms -> mnist_size * (min_zoom **<depth_level>)
    MIN_ZOOM = 2
    # Number of Glimpses
    NGLIMPSES = 7
    # Standard Deviation of the Location Policy
    LOC_STD = 0.11
    # This variable basically outlines how far (in pixels) near the borders
    # the center of each glimpse can reach with respect to the center.
    # So a value of 13 (the default) means that the center of the glimpse
    # can be anywhere between the 2rd and 27th pixel (for a 1x28x28 MNIST example).
    # So glimpses of the corner will have fewer zero-padding values
    # then if UNIT_PIXELS = 14
    UNIT_PIXELS = 26
    # Translated MNIST
    TRANSLATE = True
    # Size of each image: MNIST_SIZE x MNIST_SIZE
    TRANSLATED_MNIST_SIZE = 60

class PARAMETERS:
    """
    Class for specifying the parameters for
    the learning algorithm
    """

    #   =========================
    #   General parameters for the
    #   experiment
    #   =========================

    #   Number of learning steps
    MAX_STEPS = 1000000
    #   Number of times, the current
    #   Policy should be avaluated
    NUM_POLICY_CHECKS = 10
    #   Batch size
    BATCH_SIZE = 20

    #   =========================
    #   Algorithm specific parameters
    #   =========================

    #   To be used optimizer:
    #   rmsprop
    #   adam
    #   adadelta
    #   sgd
    OPTIMIZER = 'sgd'
    # Learning rate alpha
    LEARNING_RATE = 0.01
    # Number of steps the Learning rate should (linearly)
    # decay to MIN_LEARNING_RATE
    LEARNING_RATE_DECAY = 250000
    # Minimal Learning Rate
    MIN_LEARNING_RATE = 0.00001
    # Momentum
    MOMENTUM = 0.9
    #Discount factor gamma
    DISCOUNT = 0.95
    # Clipnorm
    CLIPNORM = 0 #-1
    # Clipvalue
    CLIPVALUE = 0 #-1

def main():
    params = PARAMETERS
    dom_opt = MNIST_DOMAIN_OPTIONS
    for i in range(1, 4):
        exp = Experiment(params, dom_opt, "./{0:03}".format(i) + "-results.json")
        del exp


if __name__ == '__main__':
    main()

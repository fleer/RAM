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
    SENSOR = 8
    # Number of zooms
    NZOOMS = 3
    # Minimal zoom size # zooms -> minRadius * 2**<depth_level>
    MIN_ZOOM = 4
    # Number of Glimpses
    NGLIMPSES = 6
    # Standard Deviation of the Location Policy
    LOC_STD = 0.11
    #
    UNIT_PIXELS = 13

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
    MAX_STEPS = 500000
    #   Number of times, the current
    #   Policy should be avaluated
    NUM_POLICY_CHECKS = 20
    #   Batch size
    BATCH_SIZE = 32

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
    # Learning rate decay
    LEARNING_RATE_DECAY = 1e-4
    # Momentum
    MOMENTUM = 0.9
    #Discount factor gamma
    DISCOUNT = 0.95
    # Clipnorm
    CLIPNORM = 0
    # Clipvalue
    CLIPVALUE = 0

def main():
    params = PARAMETERS
    dom_opt = MNIST_DOMAIN_OPTIONS
    for i in range(1, 11):
        exp = Experiment(params, dom_opt, "./{0:03}".format(i) + "-results.json")
        del exp


if __name__ == '__main__':
    main()

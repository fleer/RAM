"""
Experiment Class for Deep Mediated Interaction Learner

Author: Sascha Fleer
"""

import logging
import time
import os
import json

import numpy as np
import random

from MNIST_Domain import MNIST as Domain
from copy import deepcopy
from collections import defaultdict
from network import RAM
import cv2


class DOMAIN_OPTIONS:
    """
    Class for the Setup of the Domain Parameters
    """
    #   Number of states used to bound each episode (return to state 0 after)
    EPISODE_BOUND = 200
    #
    #   ================
    #   Reward constants
    #   ================
    #   Goal reward:
    GOAL_REWARD = +10.
    #   Step Reward
    STEP_REWARD = 0.


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
    MAX_STEPS = 10000000
    #   Number of times, the current
    #   Policy should be avaluated
    NUM_POLICY_CHECKS = 20
    #   Number of episodes, the current
    #   Policy should be evaluated
    #   ---> Resultas are averaged in the end
    CHECKS_PER_POLICY = 100


    #   =========================
    #   Parameters for image
    #   preprocecssing
    #   =========================

    #   Resized image for network input
    RESIZE_WIDTH = 84
    RESIZE_HEIGHT = 84
    # Number of images the should be merged to create one input-image
    BUFFER_LENGTH = 2
    # Number of images that should be used as one state
    PHI_LENGTH = 4



    #   =========================
    #   Experience Replay
    #   =========================

    #   Size of Batch Memory
    MEMORY_SIZE = 250000
    #   Number of samples that have to be
    #   gathered before the learning starts
    MEMORY_START = 25000
    #   Batch size
    BATCH_SIZE = 64
    # Batch accumulator
    #   - mean
    #   - sum
    BATCH_ACCUMULATOR = 'mean'


    #   =========================
    #   Epsilon decay for the
    #   greedy part of the learning
    #   algorithm
    #   --> Linear decay
    #   =========================

    #   Start value for epsilon
    EPSILON_START = 1.
    #   Final value for epsilon
    EPSILON_END = 0.1
    #   Number of steps, epsilon
    #   should decay
    EPSILON_DECAY = 100000


    #   =========================
    #   Algorithm specific parameters
    #   =========================

    # Reward discount factor gamma
    DISCOUNT = 0.95
    #   Update Frequency
    UPDATE_FREQUENCY = 4
    #   Number of steps, until the target-network
    #   should be updated
    FREEZE = 100
    #   To be used optimizer:
    #   rmsprop
    #   adadelta
    #   sdg
    OPTIMIZER = 'rmsprop'
    # If optimizer is 'rmsprop' or
    # 'adadelta' --> Specify Rho
    RHO = 0.95
    # If optimizer is 'rmsprop'
    # --> Specify Epsilon
    RMSPROP_EPSILON = 1e-6
    # Momentum term
    MOMENTUM = 0
    # Loss Clipping
    CLIP_LOSS = 1.
    # Learning Rate
    LEARNING_RATE = 0.0001
    #   Use Double-DQN algorithm
    # "Deep Reinforcement Learning with Double Q-Learning."
    #  Van Hasselt, Hado, Arthur Guez, and David Silver. AAAI. 2016.
    DOUBLE_Q = True
    #
    #   Use "Averaged-DQN"
    #   "Averaged-DQN: Variance Reduction and Stabilization for Deep Reinforcement Learning."
    #   Anschel, et.al  International Conference on Machine Learning. 2017.
    AVERAGED_DQN = False
    # Number of saved Q-Values
    Q_HISTORY = 10

class Experiment():
    """
    Main class, controlling the experiment
    """


    #   ================
    #   Evaluation
    #   ================

    # Plot the Episodes of the Performance Runs to a
    # png file for each step

    plot_to_file = False


    results = defaultdict(list)

    performance_log_template = '{total_steps: >6}: >>> Return={totreturn: >10.4g}, Success Rate={success: >10.4g} Average Steps={steps: >4}'

    def __init__(self, results_file="./results0.json"):

        logging.basicConfig(level=logging.INFO)


        logging.info("Saving results to: {}".format(results_file))

        self.max_steps = PARAMETERS.MAX_STEPS
        self.num_policy_checks = PARAMETERS.NUM_POLICY_CHECKS
        self.checks_per_policy = PARAMETERS.CHECKS_PER_POLICY
        self.epsilon_start = PARAMETERS.EPSILON_START
        self.epsilon_min = PARAMETERS.EPSILON_END
        self.epsilon_decay = PARAMETERS.EPSILON_DECAY
        self.start_memory = PARAMETERS.MEMORY_START
        self.memory_size = PARAMETERS.MEMORY_SIZE
        self.batch_size = PARAMETERS.BATCH_SIZE
        self.update_frequency = PARAMETERS.UPDATE_FREQUENCY
        if PARAMETERS.DOUBLE_Q or PARAMETERS.AVERAGED_DQN:
            assert PARAMETERS.DOUBLE_Q is not PARAMETERS.AVERAGED_DQN, "Either Double DQN ord Averaged DQN can be active!"
        self.double_q = PARAMETERS.DOUBLE_Q
        self.averaged_q = PARAMETERS.AVERAGED_DQN

        # Image height
        self.resized_height = PARAMETERS.RESIZE_HEIGHT
        # Image width
        self.resized_width = PARAMETERS.RESIZE_WIDTH

        # Number of images
        self.phi_length = PARAMETERS.PHI_LENGTH

        #Set Random State for the domain
        rng = np.random.RandomState()
        # Import Domain
        self.domain = Domain()
        self.domain.random_state = rng
        # Make Copy of Domain for performance runs
        self.performance_domain = deepcopy(self.domain)
        self.performance_domain.random_state = np.random.RandomState()

        # Initialze Learner
        self.agent = RAM(
            totalSensorBandwidth=self.domain.depth * self.domain.sensorBandwidth * self.domain.sensorBandwidth * self.domain.channels
        )

        #Start Experiment
        self.learn()
        self.save("./", results_file)

    def learn(self):

        total_steps = 0
        episode_number = 0

        while total_steps < self.max_steps:

            start_time = time.time()
            eps_return = 0.
            eps_steps = 0
            av_loss = []


            logging.debug("Start Episode!")

            s = self.get_state()

            a = self.agent.choose_action(self.data_set.phi(s), epsilon = self.epsilon, p_actions = p_actions)


            while not terminal and eps_steps < self.domain.EPISODE_BOUND:

                logging.debug("Step: " + str(eps_steps))
                # Act,Step
                nimg, r, terminal, np_actions = self.domain.step(a)

                # Put image in Buffer
                index = self.buffer_count % self.buffer_length
                self.screen_buffer[index, ...] = nimg
                self.buffer_count += 1
                ns = self.get_state()

                #create state trajectory

                na = self.agent.choose_action(self.data_set.phi(ns), epsilon = self.epsilon, p_actions = np_actions)

                self.data_set.add_sample(s, a, r, terminal)

                if len(self.data_set) > self.start_memory:

                    # Linear Epsilon Decay
                    self.epsilon = max(self.epsilon_min,
                                       self.epsilon - self.epsilon_rate)

                    if total_steps % self.update_frequency == 0:
                        imgs, actions, rewards, terminals = \
                            self.data_set.random_batch(self.batch_size)
                        # learning
                        if self.double_q:
                            loss = self.agent.train_double_q(imgs, actions, rewards, terminals)
                        elif self.averaged_q:
                            # TODO: Test Averaged DQN!
                            loss = self.agent.train_averaged_q(imgs, actions, rewards, terminals)
                        else:
                            loss = self.agent.train(imgs, actions, rewards, terminals)
                        av_loss.append(loss)

                s, a, p_actions = ns, na, np_actions

                total_steps += 1
                eps_steps += 1
                eps_return += r


                # Check Performance
                if total_steps % (self.max_steps / self.num_policy_checks) == 0:

                    self.performance_run(total_steps, episode_number)

            episode_number += 1

            #Log of loss
            if len(av_loss) > 0:
                if episode_number % 10 == 0:
                    logging.info("Total Steps={:d}: >>> Steps={:d}, steps/second: {:.2f}, average loss: {:.4f}".format( \
                        total_steps, eps_steps, eps_steps/(time.time()-start_time), np.mean(av_loss)))
            else:
                logging.info("Total Steps={:d}: >>> Still gathering Batch-Data: {}/{}".format( \
                    total_steps, total_steps, self.start_memory))

    def performance_run(self, total_steps, episode_number, visualize=False):
        performance_return = 0.
        performance_steps = 0.
        performance_term = 0.
        for j in xrange(self.checks_per_policy):
            p_step, p_ret, p_term = self.performance_run_episode()
            performance_return += p_ret
            performance_steps += p_step
            performance_term += p_term
        performance_return /= self.checks_per_policy
        performance_steps /= self.checks_per_policy
        performance_term /= self.checks_per_policy
        self.results['learning_steps'].append(total_steps)
        self.results["learning_episode"].append(episode_number)
        self.results["return"].append(performance_return)
        self.results["terminated"].append(performance_term)
        self.results["steps"].append(performance_steps)
        logging.info(
            self.performance_log_template.format(total_steps=total_steps,
                                                 totreturn=performance_return,
                                                 success= performance_term,
                                                 steps=performance_steps))

    def save(self, path, filename):
        """Saves the experimental results to ``results.json`` file
        """
        results_fn = os.path.join(path, filename)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(results_fn, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=True)
        f.close()

    def performance_run_episode(self):

        eps_return = 0
        eps_steps = 0

        img, terminal, p_actions = self.performance_domain.s0()

        # Fill buffer with image
        for a in range(self.buffer_length):
            index = self.buffer_count % self.buffer_length
            self.screen_buffer[index, ...] = img
            self.buffer_count += 1

        s = self.get_state()
        a = self.agent.choose_action(self.data_set.phi(s), epsilon = 0., p_actions = p_actions)
        while not terminal and eps_steps < self.domain.EPISODE_BOUND:

            # Act,Step
            nimg, r, terminal, np_actions = self.performance_domain.step(a)

            # Put image in Buffer
            index = self.buffer_count % self.buffer_length
            self.screen_buffer[index, ...] = nimg
            self.buffer_count += 1
            ns = self.get_state()

            #create state trajectory

            na = self.agent.choose_action(self.data_set.phi(ns), epsilon = 0., p_actions = np_actions)

            s, a, p_actions = ns, na, np_actions

            eps_steps += 1
            eps_return += r

        return eps_steps, eps_return, terminal

    def get_state(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])

        return cv2.resize(max_image,
                      (self.resized_width, self.resized_height),
                      interpolation=cv2.INTER_AREA)

    def __del__(self):
        self.results.clear()

def main():
    for i in range(11):
        exp = Experiment("./{0:03}".format(i) + "-results.json")
        del exp

if __name__ == '__main__':
    main()



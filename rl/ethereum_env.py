import datetime
import json
import os

import numpy as np
from rl.validators import Validator
from rl.utils.action_space import MultiAgentActionSpace
from rl.utils.observation_space import MultiAgentObservationSpace

import gym
from gym import spaces


class EthereumEnv(gym.Env):
    def __init__(self, validator_size):
        """
        Initialize your custom environment.
        Parameters:
            self. validator_size : an int.         The size of the validators
            self. validators : a list.             The list of the validators
            self. total_active_balance : a float.  The total active balance of the validators
            self. proportion_of_honest : a float.  The proportion of honest validators
            self. action_space : 0 or 1            The strategy of validators: 0: honest; 1: malicious
            self. observation_space : a dict.      The space of observations: validators
        ----------
        """
        # Set of validators
        self.validator_size = validator_size
        self.validators = []
        self.total_active_balance = 0
        self.proportion_of_honest = 0
        self.total_reward = [0 for _ in range(self.validator_size)]

        # The action space of validators: 0: honest; 1: malicious
        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(2) for _ in range(self.validator_size)])

        # The observation space of validators: the current balance of every validator
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) for _ in range(self.validator_size)]
        )

        # create the log directory if not exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # create logging file with timestamp
        self.log_file = open(
            "logs/log_" + str(datetime.datetime.now()) + ".txt", "w")
        # create the file and make it empty
        self.log_file.write("")

        self.window = None
        self.clock = None

        super(EthereumEnv, self).__init__()

    def reset(self):
        """
        Reset the environment and return an initial observation.
        Returns
        -------
        observation : numpy array
            The initial observation of the environment.
        """
        self.validators = []
        for i in range(self.validator_size):
            # if i < self.validator_size / 2:
            #     strategy = 0
            # else:
            #     strategy = 1
            strategy = np.random.randint(0, 2)
            status = 1
            current_balance = 32
            effective_balance = 32
            self.validators.append(
                Validator(strategy, status, current_balance, effective_balance))
            # print(f"validator {i} has strategy {strategy}, status {status}, current balance {current_balance}, and effective balance {effective_balance}.")

        proportion = 0
        for i in range(self.validator_size):
            proportion += (self.validators[i].strategy ==
                           0) / self.validator_size
        self.initial_honest_proportion = proportion
        self.proportion_of_honest = proportion
        # print(f"initial proportion of honest validators: {proportion}.")

        # Generate the initial value of total_active_balance
        self.total_active_balance = 32 * self.validator_size

        observation = self._get_obs()
        info = self._get_info()

        self.counter = 0

        return observation
        # return observation, info

    def step(self, action):
        """
        Take a step in the environment.
        Parameters
        ----------
        action : int
            The action to take in the environment.
        Returns
        -------
        observation : numpy array
            The new observation of the environment after taking the action.
        reward : float
            The reward obtained after taking the action.
        done : bool
            Whether the episode has ended or not.
        info : dict
            Additional information about the step.
        """
        # Update the actions (strategies) of validators
        for i in range(self.validator_size):
            self.validators[i].strategy = action[i]

        # Generate a proposer
        proposer = np.random.randint(0, self.validator_size)

        # Update the status of validators
        for i in range(self.validator_size):
            if i == proposer:
                self.validators[i].status = 0
            else:
                self.validators[i].status = 1

        rewards = [self.validators[i].get_reward(self.total_active_balance, self.proportion_of_honest, self.validators[proposer].strategy) for i in range(self.validator_size)]

        # Update the current balance of validators
        for i in range(self.validator_size):
            self.validators[i].update_balances(
                self.proportion_of_honest, self.total_active_balance, self.validators[proposer].strategy)
            
        # Update the proportion of honest validators
        proportion = 0
        for i in range(self.validator_size):
            proportion += (self.validators[i].strategy ==
                           0) / self.validator_size
        self.proportion_of_honest = proportion

        # Update total active balance
        total_active_balance = 0
        for i in range(self.validator_size):
            total_active_balance = total_active_balance + \
                self.validators[i].current_balance
        self.total_active_balance = total_active_balance

        observation = self._get_obs()

        info = self._get_info()

        terminated = False

        if self.total_active_balance <= 0:
            terminated = True

        if self.proportion_of_honest == 1:
            terminated = True
        elif self.proportion_of_honest == 0:
            terminated = True

        if self.counter >= 256:
            terminated = True

        payload = self.render()
        self.log_file.write(str(payload) + "\n")

        if terminated:
            return observation, rewards, terminated, info

        # counter increment
        self.counter += 1

        return observation, rewards, terminated, info

    def render(self):
        """
        Render the environment.
        Parameters
        ----------
        mode : str
            The mode to render the environment in.
        """
        payload = dict(
            proportion_of_honest=self.proportion_of_honest,
            rounds=self.counter,
            initial_honest_proportion=self.initial_honest_proportion
        )
        return payload

    def _get_obs(self):
        obs = []
        for i in range(self.validator_size):
            obs.append([self.validators[i].current_balance])
        return obs

    def _get_info(self):
        info = []  
        for i in range(self.validator_size):
            info.append([self.proportion_of_honest,self.total_active_balance,self.initial_honest_proportion])
        return info
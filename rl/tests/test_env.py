import gym
import pytest
from rl.ethereum_env import EthereumEnv
from pytest_cases import parametrize, fixture_ref
import numpy as np


@pytest.fixture(scope='module')
def env():
    env = EthereumEnv(validator_size=101)
    yield env
    env.close()


def test_init(env):
    assert env.validator_size == 101


@parametrize('env',
             [fixture_ref(env)])
def test_observation_space(env):
    obs = env.reset()
    print("ok observation: ", obs)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, rewards, done, _ = env.step(env.action_space.sample())
        print("not ok observation:", obs)
        assert env.observation_space.contains(obs)
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(env.observation_space.sample())
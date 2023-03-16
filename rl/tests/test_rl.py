# import os

# import matplotlib.pyplot as plt
# from rl.ethereum_env import EthereumEnv
# from rl.utils.helper_functions import SaveOnBestTrainingRewardCallback
# from stable_baselines3 import A2C
# from stable_baselines3.common import results_plotter
# from stable_baselines3.common.monitor import Monitor
# import pytest

# EPOCHS = 30

# @pytest.mark.skip()
# def test_a2c():
#     env = EthereumEnv(validator_size=101)

#     timesteps = 32 * (2 ** 8)

#     model = A2C("MultiInputPolicy", env, verbose=1)
#     model.learn(total_timesteps=timesteps,
#                 progress_bar=True)
#     model.save("models/a2c")


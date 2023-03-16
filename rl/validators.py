import gym
import numpy as np

class Validator():
    def __init__(self, strategy, status, current_balance, effective_balance) -> None:
        self.strategy = strategy
        self.status = status
        self.current_balance = current_balance
        self.effective_balance = effective_balance

    def get_reward(self, total_active_balance, proportion_of_honest, proposer_strategy) -> float:
        # get base reward
        base_reward = self.effective_balance * 16 / np.sqrt(total_active_balance)

        # get duty weight
        if self.status == 0:
            duty_weight = 1/8
        elif self.status == 1:
            duty_weight = 27/32
        else:
            raise ValueError("The status of the validator is not valid.")
        
        # if the proposer is honest, she proposes a valid block, and honest voters vote, while malicious voters do not vote (missing);
        # if the proposer is malicious, she proposes an invalid block, and honest voters do not vote (missing), while malicious voters vote;
        # if the proportion of honest is larger than 1/2, then if the proposer is malicious, the proposer will get a large penalty.
        
        if proportion_of_honest > 1/2:
            if proposer_strategy == 0: # honest proposer, a valid block
                if self.strategy == 0: # honest voters vote, all honest validators get rewards
                    reward = duty_weight * base_reward * proportion_of_honest
                else: # malicious voters do not vote, all malicious validators get penalized
                    reward =  - duty_weight * base_reward * proportion_of_honest
            else: # malicious proposer, an invalid block
                if self.strategy == 0: # honest voters misses voting, all honest validators get penalized
                    reward = - duty_weight * base_reward * proportion_of_honest
                else: # since honest proposers are larger than 1/2, the malicious proposer is detected and penalized heavily
                    if self.status == 0: # proposer
                        reward = - base_reward * proportion_of_honest
                    else: # malicious voters vote for it but no consensus is reached.
                        reward = 0
        else: # proportion_of_honest <= 1/2, malicious validaters are larger
            if proposer_strategy == 0: # honest proposer, a valid block
                if self.strategy == 0: # honest voters vote, but no consensus is reached, all honest validators get 0
                    reward = 0
                else: # malicious voters do not vote, all malicious validators get penalized
                    reward = - duty_weight * base_reward * proportion_of_honest
            else: # malicious proposer, an invalid block
                if self.strategy == 0: # honest voters misses voting, all honest validators get penalized
                    reward = - duty_weight * base_reward * (1 - proportion_of_honest)
                else: # malicious voters vote for it, all malicious validators get rewards
                    reward = duty_weight * base_reward * (1 - proportion_of_honest)
        return reward

    def update_balances(self, proportion_of_honest, total_active_balance, proposer_strategy):
        update = self.get_reward(total_active_balance, proportion_of_honest, proposer_strategy)
        self.current_balance = self.current_balance + update
        # update the effective balance
        if update > 1.25:
            self.effective_balance = self.effective_balance + 1
        elif update < - 0.5:
            self.effective_balance = self.effective_balance - 1
        else:
            pass
        return
import gym
import numpy as np

class Validator():
    def __init__(self, strategy, status, current_balance, effective_balance) -> None:
        self.strategy = strategy
        self.status = status
        self.current_balance = current_balance
        self.effective_balance = effective_balance

    def get_base_reward(self, total_active_balance) -> float:
        base_reward = self.effective_balance * 16 / np.sqrt(total_active_balance)
        return base_reward
    
    def duty_weight(self) -> float:
        # print("alpha: ", alpha)
        # when the validator play the honest strategy
        if self.strategy == 0:
            # when the validator is a proposer
            if self.status == 0:
                return 1/8
            # when the validator is a voter
            elif self.status == 1:
                return 27/32
            else:
                raise ValueError("The status of the validator is not valid.")
        # when the validator play the malicious strategy
        elif self.strategy == 1:
            # when the validator is a proposer: missing proposing
            if self.status == 0:
                return 0
            # when the validator is a voter: missing voting
            elif self.status == 1:
                return -27/32
            else:
                raise ValueError("The status of the validator is not valid.")
        else:
            raise ValueError("The strategy of the validator is not valid.")

    def update_balances(self, proportion_of_honest, total_active_balance) -> float:
        update = self.duty_weight * self.get_base_reward(total_active_balance = total_active_balance) * proportion_of_honest
        self.current_balance = self.current_balance + update
        # update the effective balance
        if update > 1.25:
            self.effective_balance = self.effective_balance + 1
        elif update < - 0.5:
            self.effective_balance = self.effective_balance - 1
        else:
            pass
        return
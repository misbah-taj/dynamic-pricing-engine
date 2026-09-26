import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PricingEnvironment(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.state = np.array(
            [500, 80, 30, 520],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array(
            [500, 80, 30, 520],
            dtype=np.float32
        )

        return self.state, {}

    def step(self, action):

        current_price, demand, inventory, competitor_price = self.state

        if action == 0:
            new_price = current_price * 0.95
        elif action == 1:
            new_price = current_price
        else:
            new_price = current_price * 1.05

        if new_price > current_price:
            new_demand = demand * 0.95
        elif new_price < current_price:
            new_demand = demand * 1.05
        else:
            new_demand = demand

        revenue = new_price * new_demand

        self.state = np.array(
            [new_price, new_demand, inventory, competitor_price],
            dtype=np.float32
        )

        return self.state, revenue, True, False, {}
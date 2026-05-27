import gymnasium

import numpy as np
from gymnasium.envs.registration import register


class ApartmentEnv(gymnasium.Env):
    def __init__(self, T: int, K: int, sigma: float = 0, seed=None):
        self.T = T
        self.K = K

        self.week = 1  # What week are we on? 0 is terminal

        self.u_actual = np.zeros((T,), np.int32)  # The utility of the weeks
        self.u_observed = np.zeros((T,), np.int32)

        self.sigma = sigma

        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "t": gymnasium.spaces.Discrete(
                    T + 1
                ),  # T+1 so that t=0 (Terminal) is included
                "U_t": gymnasium.spaces.Box(
                    0, K, shape=(1,)
                ),  # K+1 so that U=0 is included
            }
        )

        self.reset(seed=seed)

    def _get_obs(self):
        return {
            "t": self.week,
            "U_t": self.u_observed[self.week - 1] if self.week != 0 else 0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.u_actual = self.np_random.integers(1, self.K, size=self.T, endpoint=True)
        self.u_observed = self.u_actual + self.np_random.normal(
            0, self.sigma, size=self.T
        )

        self.week = 1

        obs = self._get_obs()

        return obs, {}

    def step(self, action: int):
        """_summary_

        Args:
            action (int): Action to take (0 is accept, 1 is reject)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        reward = 0

        if action == 0:
            reward = self.u_actual[self.week - 1] if self.week != 0 else 0
            self.week = 0

        else:
            if self.week == self.T:
                self.week = 0
            elif self.week != 0:
                self.week = self.week + 1

        return self._get_obs(), reward, self.week == 0, False, {}


if __name__ == "__main__":
    register(
        id="CS5180/PS0_ApartmentEnv",
        entry_point=ApartmentEnv,
    )

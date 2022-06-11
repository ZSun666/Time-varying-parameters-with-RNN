import math
import numpy as np
class simulate_data():
    def __init__(self, T) -> None:
        self.T = T

    def simulate_smooth(self):
        x = np.linspace(0, 4, self.T)
        beta_1 = 0.25 * (np.sin(np.pi * x) + 1)
        beta_2 = 0.1 * (np.cos(np.pi * x) + 1)
        beta_3 = np.zeros(self.T)
        beta = np.column_stack((beta_1, beta_2, beta_3))
        eta = np.random.normal(0, 0.1, self.T)
        x = np.random.uniform(0, 1, (self.T, 3)) * 10
        y = np.sum(np.multiply(x, beta), 1) + eta
        return beta, x, y

    def simulate_jump(self):
        x = np.linspace(0, 4, self.T)
        beta_1 = 1 * np.ones(self.T)
        beta_1[np.ceil(self.T / 2).astype(int):self.T] = 0.5
        beta_2 = 1 * np.ones(self.T)
        beta_2[0:np.ceil(self.T / 2).astype(int)] = 0
        beta_3 = np.zeros(self.T)
        beta = np.column_stack((beta_1, beta_2, beta_3))
        eta = np.random.normal(0, 0.1, self.T)
        x = np.random.uniform(0, 1, (self.T, 3)) * 10
        y = np.sum(np.multiply(x, beta), 1) + eta
        return beta, x, y


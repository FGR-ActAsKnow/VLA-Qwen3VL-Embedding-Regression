import numpy as np


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update(batch_mean, batch_var, batch_count)

    def _update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

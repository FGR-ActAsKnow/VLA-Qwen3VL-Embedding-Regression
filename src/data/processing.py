import json
import numpy as np
import torch


class ActionNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.mins = None
        self.maxs = None

    def fit(self, actions: np.ndarray):
        self.mean = actions.mean(axis=0)
        self.std = actions.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.mins = actions.min(axis=0)
        self.maxs = actions.max(axis=0)

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.mean) / self.std

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * self.std + self.mean

    def save(self, path: str):
        state = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        with open(path, "w") as f:
            json.dump(state, f)

    def load(self, path: str):
        with open(path) as f:
            state = json.load(f)
        self.mean = np.array(state["mean"])
        self.std = np.array(state["std"])

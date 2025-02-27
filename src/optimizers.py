# 📌 Implementing Optimizers from Scratch
import numpy as np

class SGD:
    """Vanilla Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
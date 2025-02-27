# 📌 Implementing Optimizers from Scratch
import numpy as np

class SGD:
    """Vanilla Stochastic Gradient Descent"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class SGDMomentum:
    """SGD with Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]

class NAG:
    """Nesterov Accelerated Gradient (NAG)"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            lookahead = params[i] + self.momentum * self.velocity[i]
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] = lookahead + self.velocity[i]

class AdaGrad:
    """AdaGrad Optimizer"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.cache[i] += grads[i] ** 2
            params[i] -= (self.lr / (np.sqrt(self.cache[i]) + self.epsilon)) * grads[i]

class RMSProp:
    """RMSProp Optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * grads[i] ** 2
            params[i] -= (self.lr / (np.sqrt(self.cache[i]) + self.epsilon)) * grads[i]



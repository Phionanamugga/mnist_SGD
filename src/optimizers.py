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

class Adam:
    """Adam Optimizer"""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat

# ✅ Testing the optimizers
if __name__ == "__main__":
    np.random.seed(42)

    # Simulated parameters and gradients
    params = [np.array([1.0, 2.0, 3.0])]
    grads = [np.array([0.1, -0.2, 0.3])]

    optimizers = {
        "SGD": SGD(learning_rate=0.01),
        "SGD with Momentum": SGDMomentum(learning_rate=0.01),
        "NAG": NAG(learning_rate=0.01),
        "AdaGrad": AdaGrad(learning_rate=0.01),
        "RMSProp": RMSProp(learning_rate=0.01),
        "Adam": Adam(learning_rate=0.01)
    }

    for name, optimizer in optimizers.items():
        test_params = [np.array([1.0, 2.0, 3.0])]  # Reset parameters
        optimizer.update(test_params, grads)
        print(f"{name} Updated Parameters: {test_params}")


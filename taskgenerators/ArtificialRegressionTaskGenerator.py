import torch
import numpy as np, random
from utils import DEVICE, Task

# -------------------------------------------------------------------
def sinwave():
    amplitude = np.random.uniform(.1, 5.)
    phase = np.random.uniform(0., np.pi)
    return lambda X: amplitude * np.sin(X - phase)
    
# -------------------------------------------------------------------
def linear():
    slope = np.random.uniform(0., 1.)
    intercept = np.random.uniform(0., 5.)
    return lambda X: slope * X + intercept

# -------------------------------------------------------------------
def quadratic():
    a, b, c = [np.random.uniform(0., 0.5) for _ in range(3)]
    return lambda X: a * X**2 + b * X + c

# -------------------------------------------------------------------
def l1norm():
    a, b, c = [np.random.uniform(0., 0.5) for _ in range(3)]
    return lambda X: a * np.abs(X - c) + b

# -------------------------------------------------------------------
def tanh():
    a, b, c = [np.random.uniform(0., 0.5) for _ in range(3)]
    return lambda X: a * np.tanh(X - c) + b

# -------------------------------------------------------------------
#def sigmoid():
#    a, b = [np.random.uniform(.1, 5) for _ in range(2)]
#    return lambda X: 1/(1 + np.exp(-a * (X - b)))

# -------------------------------------------------------------------
#def cubic():
#    a = np.random.uniform(0., .05)
#    return lambda X: a * X**3

# -------------------------------------------------------------------
class ArtificialRegressionTaskGenerator:
    def __init__(self, k, q, modes):
        self.k = k          # k_shot: number of examples in the support set
        self.q = q          # Number of examples in the query set
        self.modes = modes

    # -------------------------------------------------------------------
    # Generate a support set (k examples) and a query set (q examples) from the same task
    def batch(self, tml=None):
        all_functions = [sinwave(), linear(), quadratic(), l1norm(), tanh()]
        if tml is None: tml = random.choice(self.modes)
        func = all_functions[tml]
        
        # Sample a support set from this task
        X_sp = np.random.uniform(-5., 5., self.k).reshape(-1, 1)
        y_sp = func(X_sp)
        
        # Sample a query set from this task
        X_qr = np.linspace(-5., 5., self.q).reshape(-1, 1)
        y_qr = func(X_qr)

        # Convert these arrays to appropriate tensors and return them
        X_sp = torch.from_numpy(X_sp).to(DEVICE).float()
        y_sp = torch.from_numpy(y_sp).to(DEVICE).float()
        X_qr = torch.from_numpy(X_qr).to(DEVICE).float()
        y_qr = torch.from_numpy(y_qr).to(DEVICE).float()
        
        return Task(X_sp, y_sp, X_qr, y_qr, tml)
    



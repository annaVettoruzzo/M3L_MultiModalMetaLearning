from utils import func_call, accuracy
from . import MAML, Reptile
import torch, random

# -------------------------------------------------------------------
class MultiMAML:
    def __init__(self, models, loss_fn, lr_inner, lr_outer=0.001, adapt_steps=1, tasks_batch_size=1):
        self.mamls = { tml: MAML(model, loss_fn, lr_inner, lr_outer, adapt_steps, tasks_batch_size) for tml, model in models.items() }
        self.modes = list( self.mamls.keys() )
        
    def fit(self, tgen, steps=10000):
        for step in range(steps):
            tml = random.choice(self.modes)
            maml = self.mamls[tml]
            maml.fit(tgen, steps=1, tml=tml)
            if (step+1)%50 == 0:
                print(f"Step: {step+1}, tml: {tml}", end="\t\r")
        
        return self

# -------------------------------------------------------------------
class MultiReptile:
    def __init__(self, models, loss_fn, lr_inner, adapt_steps=10, eps=0.1):
        self.reptiles = { tml: Reptile(model, loss_fn, lr_inner, adapt_steps, eps) for tml, model in models.items() }
        self.modes = list( self.reptiles.keys() )
        
    def fit(self, tgen, steps=10000):
        for step in range(steps):
            tml = random.choice(self.modes)
            reptile = self.reptiles[tml]
            reptile.fit(tgen, steps=1, tml=tml)
            if (step+1)%50 == 0:
                print(f"Step: {step+1}, tml: {tml}", end="\t\r")
        
        return self


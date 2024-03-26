from utils import func_call, accuracy
import torch, copy
import numpy as np
from collections import defaultdict

# -------------------------------------------------------------------
class MAML:
    def __init__(self, model, loss_fn, lr_inner, lr_outer=0.001, adapt_steps=1, tasks_batch_size=1, with_te=False):
        self.model = model
        self.loss_fn = loss_fn
        self.lr_inner = lr_inner
        self.adapt_steps = adapt_steps
        self.tasks_batch_size = tasks_batch_size
        self.with_te = with_te
        
        self.theta = dict(self.model.named_parameters())
        meta_params = list(self.theta.values())
        self.optimizer = torch.optim.Adam(meta_params, lr=lr_outer) 
        
    # -------------------------------------------------------------------
    def adapt(self, params_dict, tsk):
        y_sp_pred, _ = func_call(self.model, params_dict, tsk, self.with_te)
        inner_loss = self.loss_fn(y_sp_pred, tsk.y_sp)
        
        grads = torch.autograd.grad(inner_loss, params_dict.values())
        adapted_params_dict = {name: w - self.lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}
        
        return adapted_params_dict
    
    # -------------------------------------------------------------------
    """
    Takes a support set (X, y) corresponding to a specific task, and returns the task specific 
    parameters phi (after adapting theta with GD using one or multiple adaptation steps)
    """
    def get_adapted_parameters(self, tsk):
        phi = self.adapt(self.theta, tsk)
        for _ in range(self.adapt_steps - 1):
            phi = self.adapt(phi, tsk)
        return phi
    
    # -------------------------------------------------------------------
    """
    Takes a set of training tasks and trains the model parameters (theta) using MAML.
    """
    def fit(self, tgen, steps=10000, tml=None):
        for step in range(steps):
            tot_loss = 0  # Will contain the average loss for a mini-batch of tasks
            
            for i in range(self.tasks_batch_size):
                tsk = tgen.batch(tml)
                phi = self.get_adapted_parameters(tsk)
                _, y_qr_pred = func_call(self.model, phi, tsk, self.with_te)
                loss = self.loss_fn(y_qr_pred, tsk.y_qr)
                tot_loss = tot_loss + loss
            
            tot_loss = tot_loss / self.tasks_batch_size  # Average loss (doesn't really matter)
            
            # Optimize tot_loss with respect to theta
            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()
            
            if (step+1)%50 == 0:
                print(f"Step: {step+1}, loss: {tot_loss.item():.5f}", end="\t\r")
        
        return self


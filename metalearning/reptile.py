import torch, copy
from utils import func_call


class Reptile:
    def __init__(self, model, loss_fn, lr_inner, adapt_steps=10, eps=0.1):
        self.model = model                      # An instance of SimpleNN()
        self.loss_fn = loss_fn                  # An instance of SimpleNN()
        self.adapt_steps = max(2, adapt_steps)  # Number of GD adaptation steps (to get task specific parameters)
        self.eps = eps                          # 0 < epsilon << 1, for interpolation
        self.inner_opt = torch.optim.SGD(model.parameters(), lr=lr_inner)
        
    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000, tml=None):
        for step in range(steps):
            # Sample a training task (no need for separate support/query sets)
            tsk = tgen.batch(tml)
            
            # Parameters before adaptation
            theta = copy.deepcopy(self.model.state_dict())
            
            # Adapt the model to this task (using several gradient steps)
            for _ in range(self.adapt_steps):
                y_pred_sp, y_pred_qr = func_call(self.model, None, tsk)
                loss = self.loss_fn(y_pred_sp, tsk.y_sp) + self.loss_fn(y_pred_qr, tsk.y_qr)
                self.inner_opt.zero_grad()
                loss.backward()
                self.inner_opt.step()
            
            # Parameters after adaptation (i.e. task specific parameters)
            params = self.model.state_dict()
            
            # Interpolate between the meta-parameters (theta) and the task specific parameters (params)
            dico = {name: theta[name] + self.eps * (params[name] - theta[name]) for name in theta.keys()}
            self.model.load_state_dict(dico)
            
            if (step+1) % 50 == 0:
                print(f"Step: {step+1}", end="\t\r")

import torch
import numpy as np
from utils import LambdaLayer, DEVICE

# -------------------------------------------------------------------
NB_MODES_REGRESSION = 5  # FIXME: Normally this number shouldn't be hardcoded like this.

# -------------------------------------------------------------------
class TaskEncoderSimple(torch.nn.Module):
    def __init__(self, out_shapes):
        super().__init__()
        out_dims = [np.prod(shape) for shape in out_shapes]
        out_dim = np.sum(out_dims)
        
        def reshape(x):
            lst = x.split(out_dims)
            return [z.view(shape) for z, shape in zip(lst, out_shapes)]
        
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(25), 
            torch.nn.ReLU(), 
            torch.nn.Linear(25, 50), 
            torch.nn.ReLU(), 
            torch.nn.Linear(50, out_dim), 
            LambdaLayer(lambda x: reshape(x))
        )
        
    def forward(self, tsk):
        tml = torch.tensor(tsk.tml).to(DEVICE)
        tml = torch.nn.functional.one_hot(tml, num_classes=NB_MODES_REGRESSION).float()
        return self.net(tml)

# -------------------------------------------------------------------
class TaskEncoderComplex(torch.nn.Module):
    def __init__(self, out_shapes):
        super().__init__()
        out_dims = [np.prod(shape) for shape in out_shapes]
        out_dim = np.sum(out_dims)
        
        def reshape(x):
            lst = x.split(out_dims)
            return [z.view(shape) for z, shape in zip(lst, out_shapes)]
        
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(25), 
            torch.nn.ReLU(), 
            torch.nn.Linear(25, 50), 
            torch.nn.ReLU(), 
            LambdaLayer(lambda x: torch.mean(x, dim=0)), 
            torch.nn.Linear(50, out_dim), 
            LambdaLayer(lambda x: reshape(x))
        )
        
    def forward(self, tsk):
        x = torch.cat((tsk.X_sp, tsk.y_sp), dim=1)
        x = self.net(x)
        return x

# -------------------------------------------------------------------
class TaskEncoderBoth(torch.nn.Module):
    def __init__(self, out_shapes):
        super().__init__()
        out_dims = [np.prod(shape) for shape in out_shapes]
        out_dim = np.sum(out_dims)
        
        def reshape(x):
            lst = x.split(out_dims)
            return [z.view(shape) for z, shape in zip(lst, out_shapes)]
        
        self.net1 = torch.nn.Sequential(
            torch.nn.LazyLinear(25), 
            torch.nn.ReLU(), 
            torch.nn.Linear(25, 50), 
            torch.nn.ReLU(), 
            LambdaLayer(lambda x: torch.mean(x, dim=0)), 
        )
        
        self.net2 = torch.nn.Sequential(
            torch.nn.LazyLinear(25), 
            torch.nn.ReLU(), 
            torch.nn.Linear(25, 50), 
            torch.nn.ReLU(), 
        )
        
        self.net3 = torch.nn.Sequential(
            torch.nn.LazyLinear(50), 
            torch.nn.ReLU(), 
            torch.nn.Linear(50, out_dim), 
            LambdaLayer(lambda x: reshape(x)), 
        )
        
    def forward(self, tsk):
        x1 = torch.cat((tsk.X_sp, tsk.y_sp), dim=1)
        x1 = self.net1(x1)
        
        tml = torch.tensor(tsk.tml).to(DEVICE)
        x2 = torch.nn.functional.one_hot(tml, num_classes=NB_MODES_REGRESSION).float()
        x2 = self.net2(x2)
        
        x = torch.cat((x1, x2), dim=-1)
        x = self.net3(x)
        
        return x

# -------------------------------------------------------------------
class SimpleRegressionModuleWithTE(torch.nn.Module):
    def __init__(self, modulation):
        super().__init__()
        
        self.modulation = modulation
        if   modulation in ["s0", "s1"]: self.te = TaskEncoderSimple(out_shapes=[(1, 25), (1, 50)])
        elif modulation in ["s2"]:       self.te = TaskEncoderSimple(out_shapes=[(2, 25), (2, 50)])
        elif modulation in ["c0", "c1"]: self.te = TaskEncoderComplex(out_shapes=[(1, 25), (1, 50)])
        elif modulation in ["c2"]:       self.te = TaskEncoderComplex(out_shapes=[(2, 25), (2, 50)])
        elif modulation in ["b0", "b1"]: self.te = TaskEncoderBoth(out_shapes=[(1, 25), (1, 50)])
        elif modulation in ["b2"]:       self.te = TaskEncoderBoth(out_shapes=[(2, 25), (2, 50)])
        
        self.block1 = self.block(1, 25)
        self.block2 = self.block(25, 50)
        self.lin = torch.nn.Linear(50, 1)
        
    def block(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), 
            torch.nn.BatchNorm1d(dim_out, track_running_stats=False), 
            torch.nn.ReLU(),
        )
    
    def modulate(self, x, z):
        if   self.modulation in ["s0", "c0", "b0"]: return x + z
        elif self.modulation in ["s1", "c1", "b1"]: return x * torch.sigmoid(z)
        elif self.modulation in ["s2", "c2", "b2"]: return x * z[0] + z[1]
    
    def forward(self, x, tsk):
        z1, z2 = self.te(tsk)
        
        x = self.block1(x)
        x = self.modulate(x, z1)
        x = self.block2(x)
        x = self.modulate(x, z2)
        x = self.lin(x)
        return x

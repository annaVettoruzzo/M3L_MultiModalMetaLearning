import torch
import numpy as np
from utils import LambdaLayer, DEVICE

# -------------------------------------------------------------------
NB_MODES_CLASSIFICATION = 8  # FIXME: Normally this number shouldn't be hardcoded like this.

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
            torch.nn.LazyLinear(100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 200), 
            torch.nn.ReLU(), 
            torch.nn.Linear(200, 200), 
            torch.nn.ReLU(), 
            torch.nn.Linear(200, out_dim), 
            LambdaLayer(lambda x: reshape(x))
        )
        
    def forward(self, tsk):
        tml = torch.tensor(tsk.tml).to(DEVICE)
        tml = torch.nn.functional.one_hot(tml, num_classes=NB_MODES_CLASSIFICATION).float()
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
        
        self.cnn_block1 = self.cnn_block(3, 64)
        self.cnn_block2 = self.cnn_block(64, 64)
        self.cnn_block3 = self.cnn_block(64, 64)
        self.flat = torch.nn.Flatten()
        
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 200), 
            torch.nn.ReLU(), 
            LambdaLayer(lambda x: torch.mean(x, dim=0)), 
            torch.nn.Linear(200, 200), 
            torch.nn.ReLU(), 
            torch.nn.Linear(200, out_dim), 
            LambdaLayer(lambda x: reshape(x))
        )
        
        
    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"), 
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2, 2), 
        )
        
    def forward(self, tsk):
        x = torch.cat((tsk.X_sp, tsk.X_qr), dim=0)
        
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.flat(x)
        
        # y = torch.nn.functional.one_hot(tsk.y_sp)
        # x = torch.cat((x, y), dim=1)
        
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
        
        self.cnn_block1 = self.cnn_block(3, 64)
        self.cnn_block2 = self.cnn_block(64, 64)
        self.cnn_block3 = self.cnn_block(64, 64)
        self.flat = torch.nn.Flatten()
        
        self.net1 = torch.nn.Sequential(
            torch.nn.LazyLinear(100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 200), 
            torch.nn.ReLU(), 
            LambdaLayer(lambda x: torch.mean(x, dim=0)), 
            torch.nn.Linear(200, 200), 
            torch.nn.ReLU(), 
        )
        
        self.net2 = torch.nn.Sequential(
            torch.nn.LazyLinear(100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 100), 
            torch.nn.ReLU(), 
            torch.nn.Linear(100, 200), 
            torch.nn.ReLU(), 
            torch.nn.Linear(200, 200), 
            torch.nn.ReLU(), 
        )
        
        self.net3 = torch.nn.Sequential(
            torch.nn.LazyLinear(200), 
            torch.nn.ReLU(), 
            torch.nn.Linear(200, out_dim), 
            LambdaLayer(lambda x: reshape(x)), 
        )
        
    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"), 
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2, 2), 
        )
        
    def forward(self, tsk):
        x1 = torch.cat((tsk.X_sp, tsk.X_qr), dim=0)
        x1 = self.cnn_block1(x1)
        x1 = self.cnn_block2(x1)
        x1 = self.cnn_block3(x1)
        x1 = self.flat(x1)
        x1 = self.net1(x1)
        
        
        tml = torch.tensor(tsk.tml).to(DEVICE)
        x2 = torch.nn.functional.one_hot(tml, num_classes=NB_MODES_CLASSIFICATION).float()
        x2 = self.net2(x2)
        
        x = torch.cat((x1, x2), dim=-1)
        x = self.net3(x)
        
        return x


# -------------------------------------------------------------------
class SimpleCNNModuleWithTE(torch.nn.Module):
    def __init__(self, n_classes, modulation):
        super().__init__()
        
        self.modulation = modulation
        self.modulation = modulation
        if   modulation in ["s0", "s1"]: self.te = TaskEncoderSimple(out_shapes=[(1, 64, 1, 1), (1, 64, 1, 1), (1, 64, 1, 1), (1, 1024), (1, 576), (1, 576)])
        elif modulation in ["s2"]:       self.te = TaskEncoderSimple(out_shapes=[(2, 64, 1, 1), (2, 64, 1, 1), (2, 64, 1, 1), (2, 1024), (2, 576), (2, 576)])
        elif modulation in ["c0", "c1"]: self.te = TaskEncoderComplex(out_shapes=[(1, 64, 1, 1), (1, 64, 1, 1), (1, 64, 1, 1), (1, 1024), (1, 576), (1, 576)])
        elif modulation in ["c2"]:       self.te = TaskEncoderComplex(out_shapes=[(2, 64, 1, 1), (2, 64, 1, 1), (2, 64, 1, 1), (2, 1024), (2, 576), (2, 576)])
        elif modulation in ["b0", "b1"]: self.te = TaskEncoderBoth(out_shapes=[(1, 64, 1, 1), (1, 64, 1, 1), (1, 64, 1, 1), (1, 1024), (1, 576), (1, 576)])
        elif modulation in ["b2"]:       self.te = TaskEncoderBoth(out_shapes=[(2, 64, 1, 1), (2, 64, 1, 1), (2, 64, 1, 1), (2, 1024), (2, 576), (2, 576)])

        
        self.cnn_block1 = self.cnn_block(3, 64)
        self.cnn_block2 = self.cnn_block(64, 64)
        self.cnn_block3 = self.cnn_block(64, 64)
        self.flat = torch.nn.Flatten()
        self.dense_block1 = self.dense_block(1024, 576)
        self.dense_block2 = self.dense_block(576, 576)
        self.lin = torch.nn.Linear(576, n_classes)
        
    def cnn_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"), 
            torch.nn.BatchNorm2d(out_channels, track_running_stats=False), 
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2, 2), 
        )
        
    def dense_block(self, dim_in, dim_out):
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
        r1, r2, r3, z0, z1, z2 = self.te(tsk)
        
        x = self.cnn_block1(x)
        x = self.modulate(x, r1)
        
        x = self.cnn_block2(x)
        x = self.modulate(x, r2)
        
        x = self.cnn_block3(x)
        x = self.modulate(x, r3)
        
        x = self.flat(x)
        x = self.modulate(x, z0)
        
        x = self.dense_block1(x)
        x = self.modulate(x, z1)
        
        x = self.dense_block2(x)
        x = self.modulate(x, z2)
        
        x = self.lin(x)
        return x


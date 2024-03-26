import torch


class SimpleRegressionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block1 = self.block(1, 25)
        self.block2 = self.block(25, 50)
        self.lin = torch.nn.Linear(50, 1)
        
    def block(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), 
            torch.nn.BatchNorm1d(dim_out, track_running_stats=False), 
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.lin(x)
        return x

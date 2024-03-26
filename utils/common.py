import torch, os, numpy as np
from tfrecord.torch.dataset import TFRecordDataset # pip install tfrecord
from cv2 import imdecode
#from torch.nn.utils.stateless import functional_call
from .stateless import functional_call
from sklearn.metrics import accuracy_score
from collections import namedtuple

# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number =  int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(gpu_number)

# -------------------------------------------------------------------
Task = namedtuple("Task", ["X_sp", "y_sp", "X_qr", "y_qr", "tml"])

# -------------------------------------------------------------------
def interpolate(d1, d2, eps=0.1):
    return {k: d1[k] + eps * (d2[k] - d1[k]) for k in d1.keys()}

# -------------------------------------------------------------------
def load_tfrecord_images(fpath):
    dataset = TFRecordDataset(fpath, None, {"image": "byte", "label": "int"})
    dataset = list(dataset)
    label = dataset[0]["label"][0]
    images = [imdecode(dico["image"], -1) for dico in dataset]
    return images, label

# -------------------------------------------------------------------
def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)

# -------------------------------------------------------------------
def func_call(model, params_dict, tsk, with_te=False):
    if params_dict is None: params_dict = dict(model.named_parameters())
    X = torch.cat((tsk.X_sp, tsk.X_qr))
    args = (X, tsk) if with_te else (X, )
    y = functional_call(model, params_dict, args)
    return y[:len(tsk.X_sp)], y[len(tsk.X_sp):]

# -------------------------------------------------------------------
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

# -------------------------------------------------------------------
def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

# -------------------------------------------------------------------
def weight_init(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)
        module.bias.data.zero_()



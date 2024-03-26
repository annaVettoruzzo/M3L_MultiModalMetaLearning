from torchvision import transforms as tr
import random
from .BaseTaskGenerator import BaseTaskGenerator
from utils import Task

# -------------------------------------------------------------------
default_transforms = tr.Compose([ 
    tr.ToPILImage(), 
    tr.Resize((32, 32)),
    #tr.Resize((28, 28)), 
    tr.ToTensor(), 
])

# -------------------------------------------------------------------
omniglot_transforms = tr.Compose([ 
    tr.ToPILImage(), 
    tr.Grayscale(num_output_channels=3), 
    tr.Resize((32, 32)),
    #tr.Resize((28, 28)), 
    tr.ToTensor(), 
    tr.Lambda(lambda v: 1. - v), # To be like the original omniglot dataset (black letter on white background)
])

# -------------------------------------------------------------------
special_transforms = tr.Compose([ 
    tr.Resize((32, 32)),
    #tr.Resize((28, 28)), 
    tr.ToTensor(), 
])

# -------------------------------------------------------------------
class AircraftTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/aircraft/"
        super().__init__(folder, n, k, q, background, transforms=default_transforms)

# -------------------------------------------------------------------
class CUBirdsTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/cu_birds/"
        super().__init__( folder, n, k, q, background, transforms=default_transforms)

# -------------------------------------------------------------------
class DtdTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/dtd/"
        super().__init__(folder, n, k, q, background, transforms=default_transforms)

# -------------------------------------------------------------------
class TrafficSignTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/traffic_sign/"
        super().__init__(folder, n, k, q, background, transforms=default_transforms)

# -------------------------------------------------------------------
class VggFlowerTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/vgg_flower/"
        super().__init__(folder, n, k, q, background, transforms=default_transforms)

# -------------------------------------------------------------------
class OmniglotTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/metadataset-records/omniglot/"
        super().__init__(folder, n, k, q, background, transforms=omniglot_transforms)
        
# -------------------------------------------------------------------
class MiniImageNetTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/miniimagenet/"
        super().__init__(folder, n, k, q, background, transforms=special_transforms)

# -------------------------------------------------------------------
class CifarTaskGenerator(BaseTaskGenerator):
    def __init__(self, n, k, q=10000, background=True):
        folder = "./data/cifarfs/"
        super().__init__(folder, n, k, q, background, transforms=special_transforms)

# -------------------------------------------------------------------
class ImageClassificationTaskGenerator:
    def __init__(self, n, k, q=10000, background=True, datasets=["aircraft", "cu_birds", "dtd", "traffic_sign", "vgg_flower", "omniglot", "miniimagenet", "cifar"]):
        dico = {
            "aircraft": AircraftTaskGenerator, 
            "cu_birds": CUBirdsTaskGenerator, 
            "dtd": DtdTaskGenerator, 
            "traffic_sign": TrafficSignTaskGenerator, 
            "vgg_flower": VggFlowerTaskGenerator, 
            "omniglot": OmniglotTaskGenerator, 
            "miniimagenet": MiniImageNetTaskGenerator, 
            "cifar": CifarTaskGenerator, 
        }
        
        self.all_gens = [TaskGenerator(n, k, q, background) for name, TaskGenerator in dico.items() if name in datasets]
        self.modes = list( range( len(self.all_gens) ) )
        
    # -------------------------------------------------------------------
    def batch(self, tml=None):
        if tml is None: tml = random.choice(self.modes) # task mode label
        tgen = self.all_gens[tml]
        X_sp, y_sp, X_qr, y_qr, _ = tgen.batch()
        return Task(X_sp, y_sp, X_qr, y_qr, tml)

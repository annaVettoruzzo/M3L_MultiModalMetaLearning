import torch, random
import numpy as np
from utils import DEVICE, Task, load_tfrecord_images
from torchvision import transforms as tr
import torchvision.datasets as datasets
from collections import defaultdict
from glob import glob
import os

# -------------------------------------------------------------------
class BaseTaskGenerator:
    def __init__(self, folder, n, k, q=10000, background=True, p_split=0.7, transforms=tr.Compose([])):
        self.n = n  # n_ways: numbre of classes
        self.k = k  # k_shot: number of examples per class in the support set
        self.q = q  # Number of examples per class in the query set
        
        if 'imagenet' in folder or 'cifar' in folder:
            split = "train" if background else "test"
            data = datasets.ImageFolder(f"{folder}/{split}", transform = transforms)
            
            # Group images by their class
            self.ds_dict = defaultdict(list)
            for img, c in data:
                self.ds_dict[c].append(img.numpy())
            # To ensure q does not exceed the number of images available per class
            for c in self.ds_dict.keys(): self.q = min(self.q, len(self.ds_dict[c]) - self.k) 
        else:
            # List all the tfrecords filenames
            fnames = glob( os.path.join(folder, "*.tfrecords") )
            nb = int(len(fnames) * p_split)
            fnames = sorted(fnames)
            fnames = fnames[:nb] if background else fnames[nb:]

            # Group images by their class
            self.ds_dict = defaultdict(list)
            for fname in fnames:
                images, c = load_tfrecord_images(fname)
                images = [transforms(img).numpy() for img in images]
                self.ds_dict[c] = images
                self.q = min(self.q, len(self.ds_dict[c]) - self.k) # To ensure q does not exceed the number of images available per class
                
        print(folder)
        print("q ==>", self.q)
    
    # -------------------------------------------------------------------
    # Sample a support set (n*k examples) and a query set (n*q examples)
    def batch(self):
        classes = list(self.ds_dict.keys())       # All possible classes
        classes = random.sample(classes, self.n)  # Randomly select n classes
        
        # Randomly map each selected class to a label in {0, ..., n-1}
        labels = random.sample(range(self.n), self.n)
        label_map = dict(zip(classes, labels))
        
        # Randomly select k support examples and q query examples from each of the selected classes
        X_sp, y_sp, X_qr, y_qr = [], [], [], []
        for c in classes:
            images = random.sample(self.ds_dict[c], self.k + self.q)
            X_sp += images[:self.k]
            y_sp += [label_map[c] for _ in range(self.k)]
            X_qr += images[self.k:]
            y_qr += [label_map[c] for _ in range(self.q)]
        
        # Transform these lists to appropriate tensors and return them
        X_sp, y_sp, X_qr, y_qr = [torch.from_numpy(np.array(lst)).to(DEVICE).float() for lst in [X_sp, y_sp, X_qr, y_qr]]
        y_sp, y_qr = y_sp.long(), y_qr.long()
        
        return Task(X_sp, y_sp, X_qr, y_qr, None)
    
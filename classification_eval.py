import warnings
warnings.filterwarnings("ignore")

from utils import DEVICE, plot_evaluation_results, plot_te_output
from utils import evaluate_classification_seeds
from taskgenerators import ImageClassificationTaskGenerator
from pathlib import Path

import torch, random
import numpy as np

from modules import SimpleCNNModuleWithTE

torch.random.manual_seed(1)
np.random.seed(1)
random.seed(1)

########################### PARAMETERS ###########################
loss_fn = torch.nn.CrossEntropyLoss()
lr_inner = 0.08
k, n = 5, 5
steps = 20
nb_test_tasks = 100


folders = [
    #f"_saved_models/classification/{k}shot-{n}way/seed0",
    f"_saved_models/classification/{k}shot-{n}way/seed10",
    f"_saved_models/classification/{k}shot-{n}way/seed100",
    f"_saved_models/classification/{k}shot-{n}way/seed1000",
    f"_saved_models/classification/{k}shot-{n}way/seed10000",
]

PATH = Path(f"_saved_models/classification/{k}shot-{n}way/")
PATH.mkdir(parents=True, exist_ok=True)

########################### DATASET ###########################
datasets_names = ["aircraft", "cu_birds", "dtd", "traffic_sign", "vgg_flower", "omniglot", "miniimagenet", "cifar"]

# background=False to sample evaluation tasks for meta-testing (not used during meta-training)
tgen = ImageClassificationTaskGenerator(n, k, background=False, datasets=datasets_names)

new_tsk = tgen.batch()
print(new_tsk.X_sp.shape)

########################### EVALUATION ###########################
dict_avg, dict_std = evaluate_classification_seeds(tgen, folders, loss_fn, lr_inner, n, steps=steps, nb_tasks=nb_test_tasks)
with open(PATH / 'new_results/results.csv', 'w') as f:
    f.write("%s,  %s,  %s,  %s,  %s\n" % ("Name", "1step", "10steps", "20steps", "std_20steps"))
    for name in dict_avg.keys():
        f.write("%s,  %.2f,  %.2f,  %.2f,  %.2f \n"%(name, dict_avg[name][1], dict_avg[name][10], dict_avg[name][-1], dict_std[name][-1]))

selected = ["scratch", "maml", "mmaml", "mmaml_kml", "multi_maml", "proposed_c1"]
dict_avg_filtered = {name: dict_avg[name] for name in selected}
dict_std_filtered = {name: dict_std[name] for name in selected}
plot_evaluation_results(dict_avg_filtered, dict_std_filtered, PATH/"new_results/results.png")

########################### EVALUATION SINGLE MODE ###########################
for lbl, mode_name in enumerate(datasets_names):
    print(f"***** tml {lbl} ==> {mode_name} *****")
    dict_avg, dict_std = evaluate_classification_seeds(tgen, folders, loss_fn, lr_inner, n, steps=steps, nb_tasks=nb_test_tasks, tml=lbl)
    with open(PATH / f'results_{mode_name}.csv', 'w') as f:
        f.write("%s,  %s,  %s,  %s,  %s  \n" % ("Name", "1step", "10steps", "20steps", "std_20steps"))
        for name in dict_avg.keys():
            f.write("%s,  %.2f,  %.2f,  %.2f,  %.2f  \n" % (name, dict_avg[name][1], dict_avg[name][10], dict_avg[name][-1], dict_std[name][-1]))

    dict_avg_filtered = {name: dict_avg[name] for name in selected}
    dict_std_filtered = {name: dict_std[name] for name in selected}
    plot_evaluation_results(dict_avg_filtered, dict_std_filtered, PATH/f"new_results/results_{mode_name}.png")
"""
########################### PLOT TE OUTPUT FOR C1 ###########################
modulation = "c1"
model = SimpleCNNModuleWithTE(n, modulation).to(DEVICE)
model.load_state_dict(torch.load(f"{folders[-2]}/proposed_{modulation}"))
df = plot_te_output(tgen, model, loss_fn, lr_inner, datasets_names, nb_test_tasks=100, steps=20, clf=True, with_te=True, save=PATH/"te_output2")

import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from utils import func_call, accuracy
import copy, numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
mse = lambda y_pred, y_true: torch.nn.MSELoss()(y_pred, y_true).cpu().detach()

# -------------------------------------------------------------------
def adapt_and_evaluate_return_model(model, loss_fn, lr_inner, tsk, steps=20, clf=False, with_te=False):
    # Copy the model to avoid adapting the original one
    cmodel = copy.deepcopy(model)

    eval_fn = accuracy if clf else mse
    optimizer = torch.optim.SGD(cmodel.parameters(), lr_inner)
    history = defaultdict(list)

    for step in range(steps + 1):
        y_sp_pred, y_qr_pred = func_call(cmodel, None, tsk, with_te)

        # Evaluate current model on the test data
        ev = eval_fn(y_qr_pred, tsk.y_qr)
        history["pred"].append(y_qr_pred.cpu().detach())
        history["eval"].append(ev)

        # Adapt the model using training data
        loss = loss_fn(y_sp_pred, tsk.y_sp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history, cmodel


# -------------------------------------------------------------------
def plot_te_output(tgen, model, loss_fn, lr_inner, datasets_names, nb_test_tasks, steps=20, clf=False, with_te=True, save=""):
    activation={}

    def get_activation(name):
        # the hook signature
        def hook(model, input, output):
            activation["te"] = output[-1][0].detach().cpu()
        return hook

    # register the vector in the last layer of the task encoder
    lst, tml_lst = [], []
    datasets_names = ["Aircraft", "CUBBirds", "DTD", "TrafficSign", "VGGFlower", "Omniglot", "MiniImagenet", "FC100"]
    for tml, name in enumerate(datasets_names):
        for _ in range(nb_test_tasks):
            tsk = tgen.batch(tml=tml)
            _, adapt_model = adapt_and_evaluate_return_model(model, loss_fn, lr_inner, tsk, steps=steps, clf=clf, with_te=with_te)
            h = adapt_model.te.register_forward_hook(get_activation("te"))
            out = adapt_model(tsk.X_qr, tsk)
            lst.append(activation["te"].numpy())
            tml_lst.append(name)
            h.remove()

    # PCA
    pca_25 = PCA(n_components=25)
    pca_result_25 = pca_25.fit_transform(np.array(lst))

    print('Cumulative explained variation for 25 principal components: {}'.format(np.sum(pca_25.explained_variance_ratio_)))

    ## TSNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=25, n_iter=1000)
    tsne_pca_results = tsne.fit_transform(pca_result_25)
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_pca_results[:, 0]
    df['tsne-2d-two'] = tsne_pca_results[:, 1]
    df['tml'] = tml_lst

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=df["tml"],
        palette=sns.color_palette("hls", len(datasets_names)),
        data=df,
        legend="full",
        alpha=0.8
    )
    if save: plt.savefig(f"{save}_tsne.png", bbox_inches='tight')
    plt.show()
    return df
"""


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
def plot_regression_tasks(tgen, n=4):
    for i in range(n):
        tsk = tgen.batch()
        plt.scatter(tsk.X_sp.cpu(), tsk.y_sp.cpu(), label=f"Task {i+1} (support)")
        plt.plot(tsk.X_qr.cpu(), tsk.y_qr.cpu(), label=f"Task {i+1} (query)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
# -------------------------------------------------------------------
def plot_regression_results(tsk, history):
    X, y, X_test, y_test = [arr.cpu().detach() for arr in [tsk.X_sp, tsk.y_sp, tsk.X_qr, tsk.y_qr]]
    losses, preds = history["eval"], history["pred"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(losses, marker='.')
    ax2.plot(X_test, y_test, label="True Function")
    ax2.plot(X, y, '^', c="r", label="Training Points")
    ax2.plot(X_test, preds[0], '--', label="After 0 steps")
    ax2.plot(X_test, preds[1], '--', label="After 1 steps")
    ax2.plot(X_test, preds[-1], '--', label=f"After {len(preds)-1} steps")
    ax1.set_xlabel("Adaptation steps (for new task)")
    ax1.set_ylabel("Test loss (for new task)")
    ax2.legend()
    plt.show()

# -------------------------------------------------------------------
def plot_classification_results(history):
    plt.plot(history["eval"], label="Test accuracy")
    plt.legend()
    plt.show()
    
# -------------------------------------------------------------------
def plot_evaluation_results(result_avg, result_std=None, save=""):
    for name in result_avg:
        avg = result_avg[name]
        plt.plot(avg, marker=".", label=name)
        
        if result_std is not None:
            std = result_std[name]
            plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=0.5)
    
    plt.xlabel("Adaptation steps (on the new task)")
    plt.ylabel("Evaluation")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show()

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
    return


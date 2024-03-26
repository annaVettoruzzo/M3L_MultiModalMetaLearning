from utils import func_call, accuracy
import torch, copy, numpy as np
from collections import defaultdict, OrderedDict
from sklearn.metrics.pairwise import euclidean_distances

# -------------------------------------------------------------------
mse = lambda y_pred, y_true: torch.nn.MSELoss()(y_pred, y_true).cpu().detach()

# -------------------------------------------------------------------
def adapt_and_evaluate(model, loss_fn, lr_inner, tsk, steps=20, clf=False, with_te=False):
    if isinstance(model, dict): 
        model = model[tsk.tml]
    
    # Copy the model to avoid adapting the original one
    cmodel = copy.deepcopy(model)
    
    eval_fn = accuracy if clf else mse
    optimizer = torch.optim.SGD(cmodel.parameters(), lr_inner)
    history = defaultdict(list)
    
    for step in range(steps+1):
        y_sp_pred, y_qr_pred = func_call(cmodel, None, tsk, with_te)
        
        # Evaluate current model on the test data
        ev = eval_fn(y_qr_pred, tsk.y_qr)
        history["pred"].append( y_qr_pred.cpu().detach() )
        history["eval"].append( ev )
        
        # Adapt the model using training data
        loss = loss_fn(y_sp_pred, tsk.y_sp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return history

# -------------------------------------------------------------------
def adapt_and_evaluate_mmaml(model, loss_fn, lr_inner, tsk, steps=20, adapt_steps=5, clf=False, with_te=False):
    # Copy the model (to avoid adapting the original one)
    cmodel = copy.deepcopy(model["model_mmaml"])
    embedding_model = model["embedding_model"]

    eval_fn = accuracy if clf else mse
    optimizer = torch.optim.SGD(cmodel.parameters(), lr_inner)
    history = defaultdict(list)

    for step in range(steps+1):
        params = OrderedDict(cmodel.named_parameters())
        embeddings = embedding_model(tsk)
        for i in range(adapt_steps):  # num_updates
            preds_sp = cmodel(tsk.X_sp, params=params, embeddings=embeddings)
            loss_sp = loss_fn(preds_sp, tsk.y_sp)
            # Update params
            grads = torch.autograd.grad(loss_sp, params.values(), create_graph=False, allow_unused=True)
            for (name, param), grad in zip(params.items(), grads):
                grad = grad.clamp(min=-20, max=20)
                if grad is not None:
                    params[name] = param - 0.001 * grad

        # Evaluate current model on the test data
        y_qr_pred = cmodel(tsk.X_qr, params=params, embeddings=embeddings)
        ev = eval_fn(y_qr_pred, tsk.y_qr)
        history["pred"].append(y_qr_pred.cpu().detach())
        history["eval"].append(ev)

        # Adapt the model using training data (support set)
        y_sp_pred = cmodel(tsk.X_sp, params=params, embeddings=embeddings)
        loss = loss_fn(y_sp_pred, tsk.y_sp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history


# -------------------------------------------------------------------
def sgd_step(model, params_dict, loss_fn, lr_inner, tsk):
    y_sp_pred, _ = func_call(model, params_dict, tsk)
    inner_loss = loss_fn(y_sp_pred, tsk.y_sp)
    if params_dict == None: params_dict = dict(model.named_parameters())
    grads = torch.autograd.grad(inner_loss, params_dict.values())
    return {name: w - lr_inner * w_grad for (name, w), w_grad in zip(params_dict.items(), grads)}


def get_adapted_parameters(model, loss_fn, lr_inner, tsk, adapt_steps):
    phi = sgd_step(model, None, loss_fn, lr_inner, tsk)
    for _ in range(adapt_steps - 1):
        phi = sgd_step(model, phi, loss_fn, lr_inner, tsk)
    return phi


def adapt_and_evaluate_tsamaml(model, loss_fn, lr_inner, tsk, steps=20, adapt_steps=5, clf=False, with_te=False):
    history = defaultdict(list)

    initial_weights = {}
    for i in range(len(model)-1):
        w = dict(model[i].named_parameters())
        initial_weights[i] = torch.cat([torch.reshape(values, [1, -1]) for _, values in w.items()], 1).cpu().detach().numpy()[0]
    initial_weights_np = np.array([weight for weight in initial_weights.values()])

    # Adaptation (get the parameters adapted for this task)
    phi_tmp = get_adapted_parameters(model["maml"], loss_fn, lr_inner, tsk, adapt_steps)
    phi_values = torch.cat([torch.reshape(values, [1, -1]) for _, values in phi_tmp.items()], 1).cpu().detach().numpy()

    # Compute distance
    dists = euclidean_distances(phi_values, initial_weights_np)
    model_idx = np.argmin(dists, -1)[0]
    task_specific_model = model[model_idx]

    # Copy the model (to avoid adapting the original one)
    cmodel = copy.deepcopy(task_specific_model)
    eval_fn = accuracy if clf else mse
    optimizer = torch.optim.SGD(cmodel.parameters(), lr_inner)

    for step in range(steps+1):
        y_sp_pred, y_qr_pred = func_call(cmodel, None, tsk)

        # Evaluate current model on the test data
        ev = eval_fn(y_qr_pred, tsk.y_qr)
        history["pred"].append(y_qr_pred.cpu().detach())
        history["eval"].append(ev)

        # Adapt the model using training data (support set)
        loss = loss_fn(y_sp_pred, tsk.y_sp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history


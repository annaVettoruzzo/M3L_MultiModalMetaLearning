from metalearning.metatest import adapt_and_evaluate, adapt_and_evaluate_mmaml, adapt_and_evaluate_tsamaml
from modules import SimpleCNNModule, SimpleCNNModuleWithTE, GatedConvModel, ConvEmbeddingModel
from utils import DEVICE
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------------
def evaluate_classification_model(tgen, name,  model, loss_fn, lr_inner, steps=20, with_te=False, nb_tasks=100, tml=None):
    lst = []
    if "mmaml" in name:
        adapt_func = adapt_and_evaluate_mmaml
    elif name == "tsa_maml":
        adapt_func = adapt_and_evaluate_tsamaml
    else:
        adapt_func = adapt_and_evaluate

    
    for _ in range(nb_tasks):
        tsk = tgen.batch(tml=tml)
        history = adapt_func(model, loss_fn, lr_inner, tsk, steps=steps, clf=True, with_te=with_te)
        lst.append(history["eval"])
    
    return np.array(lst).mean(axis=0)

# -------------------------------------------------------------------
def evaluate_classification_models(tgen, models_dict, loss_fn, lr_inner, steps=20, nb_tasks=100, tml=None):
    results = {}
    
    for name, model in models_dict.items():
        with_te = hasattr(model, 'te')
        res = evaluate_classification_model(tgen, name, model, loss_fn, lr_inner, steps=steps, with_te=with_te, nb_tasks=nb_tasks, tml=tml)
        results[name] = res
        print(f"{name}:{res}")
        
    return results

# -------------------------------------------------------------------
def load_classification_models(dirpath, n_classes):
    PATH = Path(dirpath)
    Module = lambda: SimpleCNNModule(n_classes).to(DEVICE)
    ModuleWithTE = lambda modulation: SimpleCNNModuleWithTE(n_classes, modulation).to(DEVICE)
    
    models_dict = {
        "scratch":        Module(),
        #"reptile":        Module(),
        #"multi_reptile":  defaultdict(Module),
        "maml":           Module(),
        "multi_maml":     defaultdict(Module),
        "mmaml":          dict(),
        "mmaml_kml":      dict(),
        "tsa_maml":       defaultdict(Module),
        #"proposed_s0":    ModuleWithTE("s0"),
        #"proposed_s1":    ModuleWithTE("s1"),
        #"proposed_s2":    ModuleWithTE("s2"),
        #"proposed_c0":    ModuleWithTE("c0"),
        "proposed_c1":    ModuleWithTE("c1"),
        #"proposed_c2":    ModuleWithTE("c2"),
        #"proposed_b0":    ModuleWithTE("b0"),
        #"proposed_b1":    ModuleWithTE("b1"),
        #"proposed_b2":    ModuleWithTE("b2"),
    }
    
    delete = []
    for name, model in models_dict.items():
        try:
            if isinstance(model, dict):
                dico = torch.load(PATH / name)
                if "mmaml" in name:  # FIXME:add this in models_dict
                    if "kml" in name:
                        model["model_mmaml"] = GatedConvModel(kml=True).to(DEVICE)
                        model["embedding_model"] = ConvEmbeddingModel(embedding_dims=[64, 27, 64, 4096, 9, 64, 4096, 9, 64, 4096, 9, 64], num_channels=32).to(DEVICE)
                    else:
                        model["model_mmaml"] = GatedConvModel().to(DEVICE)
                        model["embedding_model"] = ConvEmbeddingModel().to(DEVICE)

                    model["model_mmaml"].load_state_dict(dico.get("model_mmaml"))
                    model["embedding_model"].load_state_dict(dico.get("embedding_model"))
                else:
                    for tml, params in dico.items():
                        model[tml].load_state_dict(params)
            else:
                params = torch.load(PATH / name)
                model.load_state_dict(params)
        except FileNotFoundError as err:
            print(err)
            delete.append(name)
    
    for k in delete: del models_dict[k]
    return models_dict
    
# -------------------------------------------------------------------
def evaluate_classification_seeds(tgen, folders, loss_fn, lr_inner, n_classes, steps=20, nb_tasks=100, tml=None):
    final_results = defaultdict(list)
    
    for dirpath in folders:
        models_dict = load_classification_models(dirpath, n_classes)
        results = evaluate_classification_models(tgen, models_dict, loss_fn, lr_inner, steps=steps, nb_tasks=nb_tasks, tml=tml)
        for k, v in results.items():
            final_results[k].append(v)
    
    result_avg = { k: np.array(v).mean(axis=0) for k, v in final_results.items() }
    result_std = { k: np.array(v).std(axis=0) for k, v in final_results.items() }
    return result_avg, result_std    

import warnings
warnings.filterwarnings("ignore")

from metalearning import MAML, Reptile, MultiMAML, MultiReptile, MMAML, TSA_MAML
from taskgenerators import ArtificialRegressionTaskGenerator
from modules import SimpleRegressionModule, SimpleRegressionModuleWithTE, GatedNet, LSTMEmbeddingModel
from utils import DEVICE

import torch, random
import numpy as np
from pathlib import Path

# For reproducibility
seed = 0
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

########################### PARAMETERS ###########################
k, q = 5, 500
id_modes = [0, 1, 2, 3, 4]
loss_fn = torch.nn.MSELoss()
lr_inner = 0.005
meta_train_steps = 60000

# For saving models
PATH = Path(f"_saved_models/regression/{k}shot/seed{seed}")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)
########################### DATASET ###########################
tgen = ArtificialRegressionTaskGenerator(k=k, q=q, modes=id_modes)

########################### META-TRAIN ###########################
print("META-TRAIN")

print("Conditioning with sum")
model_proposed_s0 = SimpleRegressionModuleWithTE(modulation="s0").to(DEVICE)
MAML(model_proposed_s0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s0.state_dict(), PATH / "proposed_s0")

model_proposed_c0 = SimpleRegressionModuleWithTE(modulation="c0").to(DEVICE)
maml_proposed_c0 = MAML(model_proposed_c0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c0.state_dict(), PATH / "proposed_c0")

model_proposed_b0 = SimpleRegressionModuleWithTE(modulation="b0").to(DEVICE)
maml_proposed_b0 = MAML(model_proposed_b0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b0.state_dict(), PATH / "proposed_b0")

print("Conditioning with sigmoid")
model_proposed_s1 = SimpleRegressionModuleWithTE(modulation="s1").to(DEVICE)
maml_proposed_s1 = MAML(model_proposed_s1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s1.state_dict(), PATH / "proposed_s1")

model_proposed_c1 = SimpleRegressionModuleWithTE(modulation="c1").to(DEVICE)
maml_proposed_c1 = MAML(model_proposed_c1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c1.state_dict(), PATH / "proposed_c1")

model_proposed_b1 = SimpleRegressionModuleWithTE(modulation="b1").to(DEVICE)
maml_proposed_b1 = MAML(model_proposed_b1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b1.state_dict(), PATH / "proposed_b1")

print("Conditioning with FiLM")
model_proposed_s2 = SimpleRegressionModuleWithTE(modulation="s2").to(DEVICE)
maml_proposed_s2 = MAML(model_proposed_s2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s2.state_dict(), PATH / "proposed_s2")

model_proposed_c2 = SimpleRegressionModuleWithTE(modulation="c2").to(DEVICE)
maml_proposed_c2 = MAML(model_proposed_c2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c2.state_dict(), PATH / "mproposed_c2")

model_proposed_b2 = SimpleRegressionModuleWithTE(modulation="b2").to(DEVICE)
maml_proposed_b2 = MAML(model_proposed_b2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b2.state_dict(), PATH / "proposed_b2")

print("Scratch")
model_scratch = SimpleRegressionModule().to(DEVICE)
torch.save(model_scratch.state_dict(), PATH / "scratch")

print("MAML")
model_maml = SimpleRegressionModule().to(DEVICE)
MAML(model_maml, loss_fn, lr_inner, adapt_steps=2).fit(tgen, meta_train_steps)
torch.save(model_maml.state_dict(), PATH / "maml")

model_multi_maml = { tml: SimpleRegressionModule().to(DEVICE) for tml in tgen.modes }
MultiMAML(model_multi_maml, loss_fn, lr_inner, adapt_steps=2).fit(tgen, meta_train_steps)
torch.save({tml: model.state_dict() for tml, model in model_multi_maml.items()}, PATH / "multi_maml")

print("MMAML")
model_mmaml = GatedNet(input_size=np.prod(1), output_size=1, hidden_sizes=[40, 40], condition_type="affine", condition_order="low2high").to(DEVICE)
model_parameters = list(model_mmaml.parameters())
embedding_model = LSTMEmbeddingModel(input_size=np.prod(1), output_size=1, embedding_dims=[80, 80], hidden_size=40, num_layers=2, device=DEVICE).to(DEVICE)
embedding_parameters = list(embedding_model.parameters())
optimizers = (torch.optim.Adam(model_parameters, lr=0.001), torch.optim.Adam(embedding_parameters, lr=0.001))
MMAML(model_mmaml, embedding_model, optimizers, fast_lr=lr_inner, loss_func=loss_fn, first_order=True, num_updates=8,
      inner_loop_grad_clip=10, collect_accuracies=False, device=DEVICE, embedding_grad_clip=2, model_grad_clip=2).fit(tgen, meta_train_steps)
torch.save({"model_mmaml":model_mmaml.state_dict(), "embedding_model":embedding_model.state_dict()}, PATH / "mmaml")

print("TSA-MAML")
pre_model = SimpleRegressionModule().to(DEVICE)
pre_model.load_state_dict(torch.load(PATH / "maml"))
model_list = [SimpleRegressionModule().to(DEVICE) for _ in range(len(id_modes))]  # Number of clusters = number of modes
TSA_MAML(pre_model, model_list, len(id_modes), loss_fn, lr_inner, adapt_steps=8).fit(tgen, num_tasks=5000, steps=meta_train_steps)
torch.save({idx: model.state_dict() for idx, model in enumerate(model_list)}, PATH / "tsa_maml")

print("Reptile")
model_reptile = SimpleRegressionModule().to(DEVICE)
Reptile(model_reptile, loss_fn, lr_inner, adapt_steps=10, eps=0.1).fit(tgen, meta_train_steps)
torch.save(model_reptile.state_dict(), PATH / "reptile")

model_multi_reptile = { tml: SimpleRegressionModule().to(DEVICE) for tml in tgen.modes }
MultiReptile(model_multi_reptile, loss_fn, lr_inner, adapt_steps=10, eps=0.1).fit(tgen, meta_train_steps)
torch.save({tml: model.state_dict() for tml, model in model_multi_reptile.items()}, PATH / "multi_reptile")

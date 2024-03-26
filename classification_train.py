import warnings
warnings.filterwarnings("ignore")
from metalearning import MAML, Reptile, MultiMAML, MultiReptile, MMAML, TSA_MAML
from taskgenerators import ImageClassificationTaskGenerator
from modules import SimpleCNNModule, SimpleCNNModuleWithTE, GatedConvModel, ConvEmbeddingModel
from utils import DEVICE

import torch, random
import numpy as np
from pathlib import Path

# For reproducibility
seed = 1000
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

########################### PARAMETERS ###########################
k, n = 5, 5

loss_fn = torch.nn.CrossEntropyLoss()
lr_inner = 0.08
meta_train_steps = 60000

# For saving models
PATH = Path(f"_saved_models/classification/{k}shot-{n}way/seed{seed}")
PATH.mkdir(parents=True, exist_ok=True)
print(PATH)
########################### DATASET ###########################
datasets = ["aircraft", "cu_birds", "dtd", "traffic_sign", "vgg_flower", "omniglot", "miniimagenet", "cifar"]

# To sample training tasks randomly (for meta-training); background=True
tgen = ImageClassificationTaskGenerator(n, k, background=True, datasets=datasets)
new_tsk = tgen.batch()
print(new_tsk.X_sp.shape)

########################### META-TRAIN ###########################
print("META-TRAIN")
"""
print("Conditioning with sum")
model_proposed_s0 = SimpleCNNModuleWithTE(n, modulation="s0").to(DEVICE)
maml_proposed_s0 = MAML(model_proposed_s0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s0.state_dict(), PATH / "proposed_s0")

model_proposed_c0 = SimpleCNNModuleWithTE(n, modulation="c0").to(DEVICE)
maml_proposed_c0 = MAML(model_proposed_c0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c0.state_dict(), PATH / "proposed_c0")

model_proposed_b0 = SimpleCNNModuleWithTE(n, modulation="b0").to(DEVICE)
maml_proposed_b0 = MAML(model_proposed_b0, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b0.state_dict(), PATH / "proposed_b0")

print("Conditioning with sigmoid")
model_proposed_s1 = SimpleCNNModuleWithTE(n, modulation="s1").to(DEVICE)
maml_proposed_s1 = MAML(model_proposed_s1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s1.state_dict(), PATH / "proposed_s1")

model_proposed_c1 = SimpleCNNModuleWithTE(n, modulation="c1").to(DEVICE)
maml_proposed_c1 = MAML(model_proposed_c1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c1.state_dict(), PATH / "proposed_c1")

model_proposed_b1 = SimpleCNNModuleWithTE(n, modulation="b1").to(DEVICE)
maml_proposed_b1 = MAML(model_proposed_b1, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b1.state_dict(), PATH / "proposed_b1")

print("Conditioning with FiLM")
model_proposed_s2 = SimpleCNNModuleWithTE(n, modulation="s2").to(DEVICE)
maml_proposed_s2 = MAML(model_proposed_s2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_s2.state_dict(), PATH / "proposed_s2")

model_proposed_c2 = SimpleCNNModuleWithTE(n, modulation="c2").to(DEVICE)
maml_proposed_c2 = MAML(model_proposed_c2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_c2.state_dict(), PATH / "proposed_c2")

model_proposed_b2 = SimpleCNNModuleWithTE(n, modulation="b2").to(DEVICE)
maml_proposed_b2 = MAML(model_proposed_b2, loss_fn, lr_inner, adapt_steps=2, with_te=True).fit(tgen, meta_train_steps)
torch.save(model_proposed_b2.state_dict(), PATH / "proposed_b2")

print("Scratch")
model_scratch = SimpleCNNModule(n).to(DEVICE)
torch.save(model_scratch.state_dict(), PATH / "scratch")

print("MAML")
model_maml = SimpleCNNModule(n).to(DEVICE)
maml_maml = MAML(model_maml, loss_fn, lr_inner, adapt_steps=2).fit(tgen, meta_train_steps)
torch.save(model_maml.state_dict(), PATH / "maml")

model_multi_maml = { tml: SimpleCNNModule(n).to(DEVICE) for tml in tgen.modes }
MultiMAML(model_multi_maml, loss_fn, lr_inner, adapt_steps=2).fit(tgen, meta_train_steps)
torch.save({tml: model.state_dict() for tml, model in model_multi_maml.items()}, PATH / "multi_maml")
"""
print("MMAML")
model_mmaml = GatedConvModel(input_channels=3, output_size=n, use_max_pool=False, num_channels=64, img_side_len=32, condition_type="affine", kml=False).to(DEVICE)
model_parameters = list(model_mmaml.parameters())
embedding_model = ConvEmbeddingModel(input_size=np.prod((3, 32, 32)), output_size=n, embedding_dims=[128, 128, 128, 128], hidden_size=256,
                                         num_layers=2, convolutional=True, num_conv=4, num_channels=32, rnn_aggregation=(not True), embedding_pooling="avg", batch_norm=True,
                                         avgpool_after_conv=True, linear_before_rnn=False, img_size=(3, 32, 32)).to(DEVICE)
embedding_parameters = list(embedding_model.parameters())
optimizers = (torch.optim.Adam(model_parameters, lr=0.001), torch.optim.Adam(embedding_parameters, lr=0.001))
MMAML(model_mmaml, embedding_model, optimizers, fast_lr=lr_inner, loss_func=loss_fn, first_order=False, num_updates=2,
    inner_loop_grad_clip=20, collect_accuracies=True, device=DEVICE, embedding_grad_clip=0).fit(tgen, meta_train_steps)
torch.save({"model_mmaml":model_mmaml.state_dict(), "embedding_model":embedding_model.state_dict()}, PATH / "mmaml")

print("MMAML-KML")
model_mmaml_kml = GatedConvModel(input_channels=3, output_size=n, use_max_pool=False, num_channels=64, img_side_len=32, condition_type="affine", kml=True).to(DEVICE)
model_parameters = list(model_mmaml_kml.parameters())
embedding_model_kml = ConvEmbeddingModel(input_size=np.prod((3, 32, 32)), output_size=n, embedding_dims=[64, 27, 64, 4096, 9, 64, 4096, 9, 64, 4096, 9, 64], hidden_size=256,
                                         num_layers=2, convolutional=True, num_conv=4, num_channels=32, rnn_aggregation=(not True), embedding_pooling="avg", batch_norm=True,
                                         avgpool_after_conv=True, linear_before_rnn=False, img_size=(3, 32, 32)).to(DEVICE)
embedding_parameters = list(embedding_model_kml.parameters())
optimizers = (torch.optim.Adam(model_parameters, lr=0.001), torch.optim.Adam(embedding_parameters, lr=0.001))
MMAML(model_mmaml_kml, embedding_model_kml, optimizers, fast_lr=lr_inner, loss_func=loss_fn, first_order=False, num_updates=2,
    inner_loop_grad_clip=20, collect_accuracies=True, device=DEVICE, embedding_grad_clip=0).fit(tgen, meta_train_steps)
torch.save({"model_mmaml":model_mmaml_kml.state_dict(), "embedding_model":embedding_model_kml.state_dict()}, PATH / "mmaml_kml")

print("TRAIN TSA-MAML")
pre_model = SimpleCNNModule(n).to(DEVICE)
pre_model.load_state_dict(torch.load(PATH / "maml"))
model_list = [SimpleCNNModule(n).to(DEVICE) for _ in range(len(datasets))]
TSA_MAML(pre_model, model_list, len(datasets), loss_fn, lr_inner, adapt_steps=5).fit(tgen, num_tasks=2500, steps=meta_train_steps)
torch.save({idx: model.state_dict() for idx, model in enumerate(model_list)}, PATH / "tsa_maml")
"""
print("Reptile")
model_reptile = SimpleCNNModule(n).to(DEVICE)
model_reptile(new_tsk.X_sp) # Just so that PyTorch can initialize some lazy layers
Reptile(model_reptile, loss_fn, lr_inner, adapt_steps=10, eps=0.1).fit(tgen, meta_train_steps)
torch.save(model_reptile.state_dict(), PATH / "reptile")

model_multi_reptile = { tml: SimpleCNNModule(n).to(DEVICE) for tml in tgen.modes }
for m in model_multi_reptile.values(): m(new_tsk.X_sp) # Just so that PyTorch can initialize some lazy layers
MultiReptile(model_multi_reptile, loss_fn, lr_inner, adapt_steps=10, eps=0.1).fit(tgen, meta_train_steps)
torch.save({tml: model.state_dict() for tml, model in model_multi_reptile.items()}, PATH / "multi_reptile")

"""
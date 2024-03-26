from collections import OrderedDict
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from utils import get_grad_norm


# -------------------------------------------------------------------
class MMAML:
    def __init__(self, model, embedding_model, optimizers, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 collect_accuracies, device, embedding_grad_clip=0,
                 model_grad_clip=0):
        self._model = model
        self._embedding_model = embedding_model
        self._fast_lr = fast_lr
        self._optimizers = optimizers
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._collect_accuracies = collect_accuracies
        self._device = device
        self._embedding_grad_clip = embedding_grad_clip
        self._model_grad_clip = model_grad_clip
        self._grads_mean = []
        self.tbs = 1  #task batch size

    # -------------------------------------------------------------------
    def update_params(self, loss, params):
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
                params[name] = param - self._fast_lr * grad
        return params

    # -------------------------------------------------------------------
    def adapt(self, tgen, tml):
        adapted_params = []
        embeddings_list = []
        tot_loss = []
        for i in range(self.tbs):
            # Get a random task: sample support set and query set
            tsk = tgen.batch(tml)
            params = OrderedDict(self._model.named_parameters())
            embeddings = self._embedding_model(tsk, None)
            for i in range(self._num_updates):
                preds_sp = self._model(tsk.X_sp, params=params, embeddings=embeddings)
                loss_sp = self._loss_func(preds_sp, tsk.y_sp)
                params = self.update_params(loss_sp, params=params)

            preds_qr = self._model(tsk.X_qr, params=params, embeddings=embeddings)
            loss_qr = self._loss_func(preds_qr, tsk.y_qr)

            tot_loss.append(loss_qr)
            adapted_params.append(params)
            embeddings_list.append(embeddings)

        mean_loss = torch.mean(torch.stack(tot_loss))
        for optimizer in self._optimizers:
            optimizer.zero_grad()
        mean_loss.backward()

        self._optimizers[0].step()

        if len(self._optimizers) > 1:
            if self._embedding_grad_clip > 0:
                _grad_norm = clip_grad_norm_(self._embedding_model.parameters(), self._embedding_grad_clip)
            else:
                _grad_norm = get_grad_norm(self._embedding_model.parameters())
            # grad_norm
            self._grads_mean.append(_grad_norm)
            self._optimizers[1].step()

        return mean_loss, adapted_params, embeddings_list

    # -------------------------------------------------------------------
    def fit(self, tgen, steps=10000, tml=None):
        for step in range(steps):
            loss, adapted_params, embeddings = self.adapt(tgen, tml)

            if (step + 1) % 50 == 0:
                print(f"Step: {step + 1}, loss: {loss:.5f}", end="\t\r")

        return self


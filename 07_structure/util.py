import jax
import jax.numpy as jnp
import equinox as eqx
import wandb

import numpy as np
import json

from model import MLP


def save(model, hparams, path):
    with open(path, 'wb') as f:
        hyperparam_str = json.dumps(hparams)
        f.write((hyperparam_str + '\n').encode())
        eqx.tree_serialise_leaves(f, model)


def load(path):
    with open(path, 'rb') as f:
        hyperparam_str = f.readline().decode()
        hparams = json.loads(hyperparam_str)
        key = jax.random.PRNGKey(0)
        model = MLP(hparams, key)
        return eqx.tree_deserialise_leaves(f, model)


def dataloader(x, y, batch_size):
    indices = np.arange(x.shape[0])
    while True:
        perm = np.random.permutation(indices)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            yield x[idx], y[idx]


def create_data(dataset_size, key):
    x = jnp.linspace(0, 2 * jnp.pi, dataset_size)
    x = x.reshape((dataset_size, 1))
    epsilon = jax.random.normal(key, x.shape) * 0.01
    y = jnp.sin(x) + epsilon
    return x, y


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn):
        self.model = model
        self.scheduler = scheduler.gen_scheduler()
        self.optim = optimizer(learning_rate=self.scheduler)
        self.loss_fn = loss_fn

    @eqx.filter_jit
    def train_epoch(self, model, x, y, opt_state):
        loss, grads = self.loss_fn(model, x, y)
        updates, opt_state = self.optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train(self, epochs, dl_train, dl_val):
        model = self.model
        opt_state = self.optim.init(model)

        for epoch, (x, y), (val_x, val_y) in zip(range(epochs), dl_train, dl_val):
            loss, model, opt_state = self.train_epoch(model, x, y, opt_state)
            val_loss, _ = self.loss_fn(model, val_x, val_y)
            wandb.log({"train_loss": loss, "val_loss": val_loss, "lr": self.scheduler(epoch)})

        return model

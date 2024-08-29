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
    dataset_size = x.shape[0]
    indices = np.arange(dataset_size)
    num_batches = int(np.floor(dataset_size / batch_size))
    #while True:
    #    perm = np.random.permutation(indices)
    #    for start in range(0, len(perm), batch_size):
    #        idx = perm[start:start + batch_size]
    #        yield x[idx], y[idx]
    perm = np.random.permutation(indices)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, dataset_size)
        idx = perm[start:end]
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
        self.optim = optimizer
        self.loss_fn = loss_fn

    @eqx.filter_jit
    def train_epoch(self, model, x, y, opt_state):
        loss, grads = self.loss_fn(model, x, y)
        updates, _ = self.optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model

    def train(self, epochs, x_train, y_train, x_val, y_val, batch_size):
        model = self.model
        opt_state = self.optim.init(model)
        num_train_batches = int(np.floor(x_train.shape[0] / batch_size))
        num_val_batches = int(np.floor(x_val.shape[0] / batch_size))

        for epoch in range(epochs):
            opt_state.hyperparams['learning_rate'] = self.scheduler(epoch)
            total_train_loss = 0
            total_val_loss = 0
            dl_train = dataloader(x_train, y_train, batch_size)
            dl_val = dataloader(x_val, y_val, batch_size)

            for x, y in dl_train:
                loss, model = self.train_epoch(model, x, y, opt_state)
                total_train_loss += loss

            for x, y in dl_val:
                val_loss, _ = self.loss_fn(model, x, y)
                total_val_loss += val_loss

            loss = total_train_loss / num_train_batches
            val_loss = total_val_loss / num_val_batches
            wandb.log({"train_loss": loss, "val_loss": val_loss, "lr": self.scheduler(epoch)})
            print(f"Epoch {epoch}: train loss {loss:.4e}, val loss {val_loss:.4e}, lr {self.scheduler(epoch):.4e}")

        return model

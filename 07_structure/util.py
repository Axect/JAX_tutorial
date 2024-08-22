import jax
import jax.numpy as jnp
import equinox as eqx

from model import MLP

import numpy as np
import json


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

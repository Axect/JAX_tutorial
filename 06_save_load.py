import equinox as eqx
import jax
from jax import random, vmap, jit
import jax.numpy as jnp
import optax

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import json


def create_hyperbolic_lr_schedule(upper_bound, max_iter, init_lr, infimum_lr):
    if upper_bound < max_iter:
        raise ValueError("upper_bound must be greater than max_iter")
    elif infimum_lr >= init_lr:
        raise ValueError("infimum_lr must be less than init_lr")
    
    delta_lr = init_lr - infimum_lr

    @jit
    def lr_schedule(iteration):
        x = iteration
        N = max_iter
        U = upper_bound
        return init_lr + delta_lr * (
            jnp.sqrt((N - x) / U * (2 - (N + x) / U)) - jnp.sqrt(N / U * (2 - N / U))
        )
    
    return lr_schedule

def create_exp_hyperbolic_lr_schedule(upper_bound, max_iter, init_lr, infimum_lr):
    if upper_bound < max_iter:
        raise ValueError("upper_bound must be greater than max_iter")
    elif infimum_lr >= init_lr:
        raise ValueError("infimum_lr must be less than init_lr")
    
    lr_ratio = init_lr / infimum_lr

    @jax.jit
    def lr_schedule(iteration):
        x = iteration
        N = max_iter
        U = upper_bound
        return init_lr * lr_ratio ** (
            jnp.sqrt((N - x) / U * (2 - (N + x) / U)) - jnp.sqrt(N / U * (2 - N / U))
        )
    
    return lr_schedule


class MLP(eqx.Module):
    # layers only contain eqx.nn.Linear (do not contain activation functions - i.e. jax.nn.gelu)
    # if containing activation functions, optax can't be used to optimize the model
    layers: list 

    def __init__(self, hparams, key):
        width = hparams["width"]
        depth = hparams["depth"]

        in_size = 1
        out_size = 1

        layers = []
        keys = random.split(key, depth)
        for i in range(depth-1):
            layers.append(eqx.nn.Linear(in_size, width, key=keys[i]))
            in_size = width
        layers.append(eqx.nn.Linear(in_size, out_size, key=keys[-1]))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


def save(model, hparams, path):
    with open(path, 'wb') as f:
        hyperparam_str = json.dumps(hparams)
        f.write((hyperparam_str + '\n').encode())
        eqx.tree_serialise_leaves(f, model)


def load(path):
    with open(path, 'rb') as f:
        hyperparam_str = f.readline().decode()
        hparams = json.loads(hyperparam_str)
        key = random.PRNGKey(0)
        model = MLP(hparams, key)
        return eqx.tree_deserialise_leaves(f, model)


@jit
def loss_fn(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)


def dataloader(x, y, batch_size):
    indices = np.arange(len(x))
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= x.shape[0]:
            batch_perm = perm[start:end]
            yield x[batch_perm], y[batch_perm]
            start = end
            end += batch_size


def create_data(dataset_size, key):
    x = jnp.linspace(0, 2 * jnp.pi, dataset_size)
    x = x.reshape((dataset_size, 1))
    epsilon = random.normal(key, x.shape) * 0.01
    y = jnp.sin(x) + epsilon
    return x, y


def main(
    dataset_size=10000,
    batch_size=128,
    learning_rate=1e-2,
    epochs=1001,
    width=16,
    depth=3,
    seed=42
    ):
    np.random.seed(seed)
    data_key, model_key = random.split(random.PRNGKey(seed), 2)

    x, y = create_data(dataset_size, data_key)
    dl = dataloader(x, y, batch_size)

    hparams = {
        "width": width,
        "depth": depth
    }

    model = MLP(hparams, model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    hyperbolic_lr = create_hyperbolic_lr_schedule(
        upper_bound=1500,
        max_iter=epochs,
        init_lr=learning_rate,
        infimum_lr=1e-6
    )
    optim = optax.adamw(learning_rate=hyperbolic_lr)

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    opt_state = optim.init(model)

    for epoch, (x, y) in zip(range(epochs), dl):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss {loss}, lr {hyperbolic_lr(epoch)}")

    # Save model
    print("Saving model...")
    save(model, hparams, 'checkpoints/model.eqx')
    print("Model saved!")

    # Load model
    print("Loading model...")
    model = load('checkpoints/model.eqx')
    print("Model loaded!")
    pred_ys = vmap(model)(x)
    sorted_ics = jnp.argsort(x.ravel())
    x = x[sorted_ics]
    y = y[sorted_ics]
    pred_ys = pred_ys[sorted_ics]
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, pred_ys, 'r--')
        ax.set(xlabel="x", ylabel="y")
        fig.tight_layout()
        fig.savefig("figs/06_01_sin.png", dpi=600, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()

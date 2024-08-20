import equinox as eqx
import jax
from jax import Array, random, vmap, jit, grad
from jax.tree_util import tree_map
import jax.numpy as jnp
from jaxtyping import Float, PyTree
import optax

import numpy as np

import matplotlib.pyplot as plt
import scienceplots


class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes, key):
        keys = random.split(key, len(sizes)-1)
        layers = []
        for i, key in enumerate(keys[:-1]):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=key))
        layers.append(eqx.nn.Linear(sizes[-2], sizes[-1], key=keys[-1]))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


@jit
def loss_fn(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)


def dataloader(x, y, batch_size):
    indices = np.arange(len(x))
    np.random.seed(42)
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
    learning_rate=5e-3,
    epochs=1001,
    hidden_size=16,
    depth=3,
    seed=42
    ):
    data_key, model_key = random.split(random.PRNGKey(seed), 2)

    x, y = create_data(dataset_size, data_key)
    dl = dataloader(x, y, batch_size)

    model = MLP([1] + [hidden_size] * depth + [1], model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = vmap(model)(x)
        return jnp.mean((y - pred_y) ** 2)

    optim = optax.adamw(learning_rate)

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
            print(f"Epoch {epoch}, loss {loss}")

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
        fig.savefig("figs/04_01_sin.png", dpi=600, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()

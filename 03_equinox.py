import equinox as eqx
import jax
from jax import Array, random, vmap, jit, grad
from jax.tree_util import tree_map
import jax.numpy as jnp

import matplotlib.pyplot as plt
import scienceplots


class MLP(eqx.Module):
    layers: list
    extra_bias: Array

    def __init__(self, sizes, key):
        keys = random.split(key, len(sizes)-1)
        layers = []
        for i, key in enumerate(keys[:-1]):
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=key))
        layers.append(eqx.nn.Linear(sizes[-2], sizes[-1], key=keys[-1]))
        self.layers = layers
        self.extra_bias = jnp.ones(sizes[-1])

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x) + self.extra_bias


@jit
def loss_fn(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)


batch_size = 1000
sizes = [1, 32, 32, 1]
x_key, model_key = random.split(random.PRNGKey(0))
model = MLP(sizes, model_key)
x = random.uniform(x_key, (batch_size, 1), minval=0.0, maxval=jnp.pi * 2)
y = jnp.sin(x)

print("Baseline: ", loss_fn(model, x, y))

# Compute gradients
grads = grad(loss_fn)(model, x, y)

# Perform gradient descent
learning_rate = 0.1
for i in range(1001):
    model = tree_map(lambda m, g: m - learning_rate * g, model, grads)
    if i % 50 == 0:
        print("New model: ", loss_fn(model, x, y))
    grads = grad(loss_fn)(model, x, y)

# Plot
y_pred = vmap(model)(x)
sorted_indices = jnp.argsort(x.ravel())
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(x_sorted, y_sorted, label="Original data")
    ax.plot(x_sorted, y_pred_sorted, label="Predicted data")
    ax.legend()
    ax.set(xlabel="x", ylabel="y")
    fig.tight_layout()
    fig.savefig("figs/03_01_mlp.png", dpi=600, bbox_inches="tight")
    plt.close()

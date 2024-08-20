import equinox as eqx
import jax


class MLP(eqx.Module):
    layers: list

    def __init__(self, hparams, key):
        width = hparams["width"]
        depth = hparams["depth"]

        in_size = 1
        out_size = 1

        layers = []
        keys = jax.random.split(key, depth)
        for i in range(depth-1):
            layers.append(eqx.nn.Linear(in_size, width, key=keys[i]))
            in_size = width
        layers.append(eqx.nn.Linear(in_size, out_size, key=keys[-1]))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)

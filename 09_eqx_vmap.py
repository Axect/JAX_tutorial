import jax
import jax.numpy as jnp
import equinox as eqx

class MLP(eqx.Module):
    layers: list
    
    def __init__(self, hparams, key):
        super().__init__()
        nodes = hparams['nodes']
        num_layers = hparams['layers']
        input_size = hparams['input_size']
        output_size = hparams['output_size']
        
        keys = jax.random.split(key, num_layers + 1)
        
        self.layers = [eqx.nn.Linear(input_size, nodes, key=keys[0])]
        for i in range(1, num_layers):
            self.layers.append(eqx.nn.Linear(nodes, nodes, key=keys[i]))
        self.layers.append(eqx.nn.Linear(nodes, output_size, key=keys[-1]))
    
    def __call__(self, x):
        print(x.shape)
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


# Batch: 100
# Input: 2
X = jax.random.normal(key=jax.random.PRNGKey(0), shape=(100, 2))

hparams = {
    'nodes': 20,
    'layers': 4,
    'input_size': 2,
    'output_size': 1
}

model = MLP(hparams, jax.random.PRNGKey(1))
y = jax.vmap(model)(X)
print(y.shape)

X = jax.random.normal(key=jax.random.PRNGKey(0), shape=(100, 1))

hparams = {
    'nodes': 20,
    'layers': 4,
    'input_size': 1,
    'output_size': 2
}

model = MLP(hparams, jax.random.PRNGKey(1))
y = jax.vmap(model)(X)
print(y.shape)


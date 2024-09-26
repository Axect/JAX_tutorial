import jax
import jax.numpy as jnp
import equinox as eqx


class LSTM(eqx.Module):
    cells: list

    def __init__(self, hparams, key):
        super().__init__()

        input_size = hparams['input_size']
        hidden_size = hparams['hidden_size']
        layers = hparams['layers']

        keys = jax.random.split(key, layers + 1)

        self.cells = [eqx.nn.LSTMCell(input_size, hidden_size, key=keys[0])]
        for i in range(1, layers):
            self.cells.append(eqx.nn.LSTMCell(hidden_size, hidden_size, key=keys[i]))

    def __call__(self, xs, init_state=None):
        def scan_fn(layer_state, x):
            next_states = []
            current_input = x
            for cell, (h, c) in zip(self.cells, layer_state):
                (h, c) = cell(current_input, (h, c))
                next_states.append((h, c))
                current_input = h
            return tuple(next_states), current_input

        # init state for each layer
        if init_state is None:
            init_state = tuple((jnp.zeros(cell.hidden_size), jnp.zeros(cell.hidden_size)) 
                               for cell in self.cells)

        # scan over xs
        final_states, outputs = jax.lax.scan(scan_fn, init_state, xs)

        # stacking hidden states & cell states
        h = jnp.stack([state[0] for state in final_states], axis=0) # L x H
        c = jnp.stack([state[1] for state in final_states], axis=0) # L x H
        final_states = (h, c)

        return final_states, outputs


# Batch: 100
# Window: 10
# Input: 2
X = jax.random.normal(key=jax.random.PRNGKey(0), shape=(100, 10, 2))

hparams = {
    'input_size': 2,
    'hidden_size': 20,
    'layers': 3
}

model = LSTM(hparams, jax.random.PRNGKey(1))
(h, c), outputs = jax.vmap(model)(X)

print(h.shape)
print(c.shape)
print(outputs.shape)


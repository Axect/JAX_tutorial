from util import dataloader
import jax
import jax.numpy as jnp
import numpy as np


x = jnp.arange(0, 100, dtype=jnp.float32).reshape(-1, 1)
y = jnp.arange(0, 100, dtype=jnp.float32).reshape(-1, 1)

np.random.seed(42)

for epoch, (x_batch, y_batch) in zip(range(3), dataloader(x, y, 5)):
    print(x_batch, '\n', y_batch)

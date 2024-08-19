import jax
from jax import grad, jit, lax, random, make_jaxpr, vmap
import jax.numpy as jnp
from jax.errors import TracerBoolConversionError, NonConcreteBooleanIndexError

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import time


# 01 Pure functions
print("==========================================================")
print("Pure functions")
print("==========================================================")
g = 0.
def impure_uses_globals(x):
    return x + g
print("First call: ", jit(impure_uses_globals)(4.))
g = 10.
print("Second call: ", jit(impure_uses_globals)(5.)) # No change (since inputs are same type)
print("Third call: ", jit(impure_uses_globals)(jnp.array([5.]))) # change (different types)

g = 0.
def impure_saves_global(x):
    global g
    g = x
    return x
print("First call: ", jit(impure_saves_global)(4.))
print("Saved global: ", g) # Saved global has an internal JAX value


# 02 lax fori_loop
print("==========================================================")
print("lax fori_loop")
print("==========================================================")
array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x + array[i], 0)) # 45
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x + next(iterator), 0)) # 0 (!!!)


# 03 Update array
print("==========================================================")
print("Update array")
print("==========================================================")
jax_array = jnp.zeros((3,3), dtype=jnp.float32)

try:
    jax_array[1, :] = 1.0 # jax arrays are immutable
except TypeError as e:
    print(e)

updated_array = jax_array.at[1, :].set(1.0)
added_array = updated_array.at[1, :].add(7.)
print("Original array:\n", jax_array)
print("Updated array:\n", updated_array)
print("Added array:\n", added_array)


# 04 Out-of-bounds indexing
print("==========================================================")
print("Out-of-bounds indexing")
print("==========================================================")
x = jnp.arange(10)
print(x[10]) # Index out of bounds (not an error)
print(jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan))


# 05 Non-array inputs
print("==========================================================")
print("Non-array inputs")
print("==========================================================")
try:
    print(jnp.sum([1, 2, 3]))
except TypeError as e:
    print(e)

def permissive_sum(x):
    return jnp.sum(jnp.array(x))

x = list(range(10))
print(permissive_sum(x)) # it works, but
print(make_jaxpr(permissive_sum)(x))
print(make_jaxpr(permissive_sum)(jnp.array(x)))


# 06 JAX PRNG
print("==========================================================")
print("JAX PRNG")
print("==========================================================")
key = random.key(0)
print(key)
print(random.normal(key, shape=(1,)))
print(key) # key is unchanged
print(random.normal(key, shape=(1,)))

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    |---SPLIT --> new key   ", key)
print("             |--> new subkey", subkey, "--> normal", normal_pseudorandom)

print("old key", key)
key, subkey = random.split(key)
normal_pseudorandom = random.normal(subkey, shape=(1,))
print("    |---SPLIT --> new key   ", key)
print("             |--> new subkey", subkey, "--> normal", normal_pseudorandom)

key, *subkeys = random.split(key, 4) # one is key, the rest are subkeys
for subkey in subkeys:
    print("Normal from three subkeys:", random.normal(subkey, shape=(1,)))


# 07 Python control flow + Autograd
print("==========================================================")
print("Python control flow + Autograd")
print("==========================================================")

def f(x):
    if x < 3.:
        return 3. * x ** 2
    else:
        return -4 * x

x = jnp.linspace(0, 5, 100)
start = time.time()
dydx = [grad(f)(x_i) for x_i in x] # ok but not performative
end = time.time()
print(f"Time with Autograd (naive for loop): {end - start}")

def f_lax(x):
    return lax.cond(
        x < 3,
        lambda x: 3. * x ** 2,
        lambda x: -4. * x,
        x
    )
start = time.time()
dydx = vmap(grad(f_lax))(x)
end = time.time()
print(f"Time with Autograd (lax.cond): {end - start}")

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.plot(x, dydx)
    ax.set(xlabel="x", ylabel="dy/dx")
    fig.tight_layout()
    fig.savefig("figs/02_01_dydx.png", dpi=600, bbox_inches="tight")
    plt.close()


# 08 Python control flow + JIT
print("==========================================================")
print("Python control flow + JIT")
print("==========================================================")

def f(x):
    if x < 3: # only work for numbers not arrays -> JIT compile failed
        return 3. * x ** 2
    else:
        return -4 * x

try:
    print(jit(f)(2))
except TracerBoolConversionError as e:
    print(e)

# But with static argnums, it works
print(jit(f, static_argnums=(0,))(2))

# For argument-value dependent shapes, we should need static_argnums
def example_fun(length, val):
    return jnp.ones((length,)) * val

print(example_fun(5, 4)) # it's ok

try:
    print(jit(example_fun)(5, 4))
except TypeError as e:
    print(e)

print(jit(example_fun, static_argnums=(0,))(5, 4)) # smaller subfunction (only work for (0,))


# 09 Structured control flow primitives
print("==========================================================")
print("Structured control flow primitives")
print("==========================================================")
# lax.cond: differentiable
# lax.while_loop: fwd-mode differentiable
# lax.fori_loop: fwd-mode differentiable; fwd and rev-mode differentiable if endpoints are static.
# lax.scan: differentiable

# while_loop
init_val = 0
cond_fun = lambda x: x < 10
body_fun = lambda x: x + 1
print("x=0; while x < 10: x + 1 =", lax.while_loop(cond_fun, body_fun, init_val))

# fori_loop
init_val = 0
start = 0
stop = 10
body_fun = lambda i,x: x + i
print("x=0; for i in range(10): x + i =", lax.fori_loop(start, stop, body_fun, init_val))

# 10 Dynamic shapes
print("==========================================================")
print("Dynamic shapes")
print("==========================================================")

def nansum(x):
    mask = ~jnp.isnan(x)
    x_without_nans = x[mask] # Shape changed!
    return x_without_nans.sum()

x = jnp.array([1, 2, jnp.nan, 3, 4])
print(nansum(x)) # it works outside JIT

try:
    jit(nansum)(x) # because of dynamic shape
except NonConcreteBooleanIndexError as e:
    print(e)

@jit
def nansum_2(x):
    mask = ~jnp.isnan(x)
    return jnp.where(mask, x, 0).sum()

print(nansum_2(x)) # it works!

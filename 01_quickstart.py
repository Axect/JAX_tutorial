import jax.numpy as jnp
from jax import random, jit, grad, vmap

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import scienceplots
import time


def main():
    # 01 SELU
    x = jnp.linspace(-3.0, 3.0, 100)
    y = selu(x)

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="x", ylabel="SELU(x)")
        fig.tight_layout()
        fig.savefig("figs/01_01_selu.png", dpi=600, bbox_inches="tight")
        plt.close()

    # 02 JIT
    key = random.key(42)
    x = random.normal(key, (1_000_000,))
    time_default = measure_time(selu, x)
    selu_jit = jit(selu)
    _ = selu_jit(x)
    time_jit = measure_time(selu_jit, x)
    print(f"Time without JIT: {time_default}")
    print(f"Time with JIT: {time_jit}")
    print(f"Speedup: {time_default / time_jit}")

    # 03 Grad
    derivative_fn = jit(grad(sum_logistic))
    x = jnp.linspace(-3.0, 3.0, 100)
    dy_dx_finite = first_finite_diff(sum_logistic, x)
    dy_dx_jax = derivative_fn(x)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x, dy_dx_finite, color='darkblue', label="Finite difference", alpha=0.3)
        ax.plot(x, dy_dx_jax, color='red', label="JAX", alpha=0.3)
        ax.legend()
        ax.set(xlabel="x", ylabel="dy/dx")
        fig.tight_layout()
        fig.savefig("figs/01_03_grad.png", dpi=600, bbox_inches="tight")
        plt.close()

    # 04 vmap
    key1, key2 = random.split(key)
    print("Keys: ", key1, key2)
    mat = random.normal(key1, (150, 100))
    batched_x = random.normal(key2, (10, 100))
    assert_allclose(
        naively_batched_apply_matrix(mat, batched_x),
        vmap_batched_apply_matrix(mat, batched_x),
        atol=1E-4,
        rtol=1E-4,
    )
    time_naive = measure_time(naively_batched_apply_matrix, mat, batched_x)
    time_vmap = measure_time(vmap_batched_apply_matrix, mat, batched_x)
    print(f"Time naive: {time_naive}")
    print(f"Time vmap: {time_vmap}")
    print(f"Speedup: {time_naive / time_vmap}")
    

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


def measure_time(f, *args, **kwargs):
    start = time.time()
    f(*args, **kwargs).block_until_ready()
    end = time.time()
    return end - start


def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


def first_finite_diff(f, x, eps=1E-3):
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2.0 * eps) for v in jnp.eye(len(x))])


def apply_matrix(mat):
    return lambda x: jnp.dot(mat, x)


def naively_batched_apply_matrix(mat, v_batched):
    return jnp.stack([apply_matrix(mat)(v) for v in v_batched])


@jit
def vmap_batched_apply_matrix(mat, v_batched):
    return vmap(apply_matrix(mat))(v_batched)


if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp


def dot(x,y):
    return jnp.dot(x, y)

x = jnp.asarray([
    [2, 2, 2],
    [3, 3, 3]
])

y = jnp.asarray([
    [4, 4, 4],
    [5, 5, 5]
])

#print(dot(x, y)) # : Error

vdot = jax.vmap(dot, in_axes=[0, 0])
print(vdot(x, y))
# vdot(x, y) = [dot(x[0, :], y[0, :]), dot(x[1, :], y[1, :])]

x_mat = jnp.arange(6).reshape(3, 2)
y_vec = jnp.arange(2)
y_mat = jnp.arange(6).reshape(2, 3)

vv = lambda x, y: jnp.dot(x, y)
mv = jax.vmap(vv, (0, None), 0)
# mv(x, y) = [vv(x[0, :], y), vv(x[1, :], y), ...]
mm = jax.vmap(mv, (None, 1), 1)
# mm(x, y) = [mv(x, y[:, 0]), mv(x, y[:, 1]), ...]
#          = [
#             vv(x[0, :], y[:, 0]), vv(x[0, :], y[:, 1]), ...;
#             vv(x[1, :], y[:, 0]), vv(x[1, :], y[:, 1]), ...
#            ]

print(f"x_mat:\n {x_mat}")
print(f"y_vec: {y_vec}")
print(f"x_mat * y_vec: {mv(x_mat, y_vec)}")
print(f"y_mat:\n {y_mat}")
print(f"x_mat * y_mat:\n {mm(x_mat, y_mat)}")

A = jnp.arange(200).reshape(100, 2)
B = jnp.arange(300).reshape(150, 2)
C = jax.numpy.einsum("ij,kj->ik", A, B)
print(C.shape)

# mypy: ignore-errors

# Skip type checking for now due to a known issue of JAX
# https://github.com/google/jax/issues/1555

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp


def reflector(u: npt.NDArray[np.float32]) -> jnp.ndarray:
    # Householder reflector
    norm = np.linalg.norm(u)
    u /= norm
    H = np.eye(u.size) - 2 * u[:, np.newaxis] @ u[:, np.newaxis].T
    return jnp.asarray(H)


def rank_one(u: npt.NDArray[np.float32]) -> jnp.ndarray:
    # Just a rank-one matrix
    H = u[:, np.newaxis] @ u[:, np.newaxis].T
    return jnp.asarray(H)


def just_random(u: npt.NDArray[np.float32]) -> jnp.ndarray:
    rng = np.random.default_rng(42)  # chosen by fair dice roll.
    # guaranteed to be random.
    n = u.size
    H = rng.random((n, n))
    return jnp.asarray(H)


ground_truths = {
    "reflector": reflector,
    "rank_one": rank_one,
    "just_random": just_random,
}

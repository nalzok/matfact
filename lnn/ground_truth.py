import numpy as np
import jax.numpy as jnp


def reflector(u):
    # Householder reflector
    norm = np.linalg.norm(u)
    u /= norm
    H = np.eye(u.size) - 2 * u[:, np.newaxis] @ u[:, np.newaxis].T
    return jnp.asarray(H)

def rank_one(u):
    # Just a rank-one matrix
    H = u[:, np.newaxis] @ u[:, np.newaxis].T
    return jnp.asarray(H)

def just_random(u):
    rng = np.random.default_rng(42) # chosen by fair dice roll.
                                    # guaranteed to be random.
    n = u.size
    H = rng.random((n, n))
    return jnp.asarray(H)

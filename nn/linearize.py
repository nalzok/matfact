import jax
import jax.numpy as jnp
import numpy as np


def linearize(state, cnn, X0):
    Js = []
    output = X0
    for f in (cnn.f1, cnn.f2, cnn.f3, cnn.f4):
        output, J = jax.vmap(
            lambda tangent: jax.jvp(lambda X: cnn.apply(state.params, X, method=f), (output,), (tangent,)),
        )(jnp.eye(np.prod(output.shape)).reshape(-1, *output.shape))
        output = output[0]
        Js.append(J)

    return Js, output

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from nn.model import CNN, create_train_state
from nn.linearize import linearize


class TestLEUP(unittest.TestCase):
    def test_random(self) -> None:
        key = jax.random.PRNGKey(42)
        specimen_key, state_key = jax.random.split(key)
        specimen = jax.random.uniform(specimen_key, (28, 28, 1))
        state = create_train_state(state_key, 0, specimen)

        cnn = CNN()
        output = image = specimen

        # Calculate Jacobian matrices separately
        Js, output = linearize(state, cnn, image)
        J_x = jnp.eye(np.prod(image.shape))
        for J in Js:
            J = J.reshape((J.shape[0], np.prod(J.shape[1:]))).T
            J_x = J @ J_x

        output_step = output
        J_x_step = J_x

        # Calculate the product of all Jacobian matrices directly
        output, vjp_fun = jax.vjp(cnn.apply, state.params, image)
        _, J_x = jax.vmap(vjp_fun)(
            jnp.eye(10)
        )  # the cotangent vector space is R^10, since there are 10 logits

        output_whole = output
        J_x_whole = J_x.reshape((J_x.shape[0], -1))

        self.assertIsNone(
            np.testing.assert_allclose(output_step, output_whole, atol=1e-4)
        )
        self.assertIsNone(np.testing.assert_allclose(J_x_step, J_x_whole, atol=1e-4))

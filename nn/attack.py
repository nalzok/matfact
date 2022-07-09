"""
Projected gradient descent. Adapeted from

    https://jaxopt.github.io/stable/notebooks/deep_learning/adversarial_training.html
"""
import jax
import jax.numpy as jnp
import optax


@jax.jit
def pgd_attack(image, target, state, epsilon=0.1, maxiter=10):
    image_perturbation = jnp.zeros_like(image)

    @jax.grad
    def loss_fn(perturbation):
        logits = state.apply_fn(state.params, image + perturbation)
        loss = optax.softmax_cross_entropy(logits, target)
        return loss.sum()

    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(loss_fn(image_perturbation))
        # heuristic step-size 2 eps / maxiter
        image_perturbation -= (2 * epsilon / maxiter) * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, -epsilon, epsilon)

    return jnp.clip(image + image_perturbation, 0, 1)

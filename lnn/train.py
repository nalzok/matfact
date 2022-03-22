# mypy: ignore-errors

from itertools import islice

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from tqdm import tqdm


def data_gen(np_rng, p, H):
    while True:
        v = np_rng.normal(size=p).astype(np.float32)
        yield v, H @ v


class LNN(hk.Module):
    def __init__(self, p, name=None):
        super().__init__(name)
        self.p = p
        self.model = hk.Sequential(
            [
                hk.Linear(self.p, with_bias=False, name="linear_0"),
                hk.Linear(self.p//2, with_bias=False, name="linear_1"),
                hk.Linear(self.p, with_bias=False, name="linear_2"),
            ]
        )

    def __call__(self, input):
        output = self.model(input)
        return output


def update_rule(param, update):
    return param - 0.001 * update


def train(ground_truth) -> tuple[float, list[np.ndarray]]:
    p = 8

    def forward(input):
        lnn = LNN(p)
        output = lnn(input)
        return output

    forward_t = hk.transform(forward)
    forward_t = hk.without_apply_rng(forward_t)

    def loss_fn(input, ground_truth):
        output = forward(input)
        return jnp.linalg.norm(output - ground_truth)

    loss_fn_t = hk.transform(loss_fn)
    loss_fn_t = hk.without_apply_rng(loss_fn_t)

    np_rng = np.random.default_rng(42)
    u = np_rng.normal(size=p)
    H = ground_truth(u)
    input_dataset = data_gen(np_rng, p, H)

    _, s, _ = np.linalg.svd(H, full_matrices=False)

    dummy_u, dummy_v = next(input_dataset)
    rng = jax.random.PRNGKey(42)
    params = loss_fn_t.init(rng, dummy_u, dummy_v)

    for u, v in (pbar := tqdm(islice(input_dataset, 16384))):
        I = np.eye(p)
        Hhat = forward_t.apply(params, I)
        _, shat, _ = np.linalg.svd(Hhat, full_matrices=False)
        pbar.set_description(f"{np.round(s-shat, 2)}")

        grads = jax.grad(loss_fn_t.apply)(params, u, v)
        params = jax.tree_util.tree_multimap(update_rule, params, grads)

    I = np.eye(p)
    Hhat = forward_t.apply(params, I)
    _, shat, _ = np.linalg.svd(Hhat, full_matrices=False)
    loss = jnp.linalg.norm(s - shat)
    weights = list(np.array(params[f"lnn/~/linear_{i}"]["w"]) for i in range(3))

    return loss, weights


if __name__ == "__main__":
    from ground_truth import just_random
    train(just_random)

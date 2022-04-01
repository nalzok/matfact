# mypy: ignore-errors

from itertools import islice

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from tqdm import tqdm
import click

from lnn.ground_truth import reflector, rank_one, just_random


def data_gen(np_rng, p, H):
    while True:
        v = np_rng.normal(size=p).astype(np.float32)
        yield v, H @ v


class LNN(hk.Module):
    def __init__(self, neurons: list[int], name=None):
        super().__init__(name)
        self.neurons = neurons
        self.model = hk.Sequential(
            [
                hk.Linear(p, with_bias=False, name=f"linear_{i}")
                for i, p in enumerate(self.neurons)
            ]
        )

    def __call__(self, input):
        output = self.model(input)
        return output


def update_rule(param, update):
    return param - 0.001 * update


ground_truths = {
    "reflector": reflector,
    "rank_one": rank_one,
    "just_random": just_random,
}


def lookup_ground_truth(ctx, param, value):
    try:
        return ground_truths[value]
    except KeyError:
        raise click.BadParameter("Invalid choice of 'ground_truth'")


@click.command()
@click.option(
    "--ground_truth",
    type=click.Choice(list(ground_truths.keys()), case_sensitive=False),
    callback=lookup_ground_truth,
    help="Generator for the ground truth linear transformation H.",
)
@click.option("--neurons", type=int, multiple=True, help="Dimension of H.")
@click.option("--epochs", type=int, help="Number of training epochs.")
def cli(ground_truth, neurons, epochs):
    train(ground_truth, neurons, epochs)


def train(ground_truth, neurons, epochs) -> tuple[float, list[np.ndarray]]:
    def forward(input):
        lnn = LNN(neurons)
        output = lnn(input)
        return output

    forward_t = hk.transform(forward)
    forward_t = hk.without_apply_rng(forward_t)

    def loss_fn(input, ground_truth):
        output = forward(input)
        return jnp.linalg.norm(output - ground_truth)

    loss_fn_t = hk.transform(loss_fn)
    loss_fn_t = hk.without_apply_rng(loss_fn_t)

    p = neurons[0]
    np_rng = np.random.default_rng(42)
    u = np_rng.normal(size=p)
    H = ground_truth(u)
    input_dataset = data_gen(np_rng, p, H)

    u, s, v = np.linalg.svd(H, full_matrices=False)

    dummy_u, dummy_v = next(input_dataset)
    rng = jax.random.PRNGKey(42)
    params = loss_fn_t.init(rng, dummy_u, dummy_v)

    for u, v in (pbar := tqdm(islice(input_dataset, epochs))):
        I = np.eye(p)
        Hhat = forward_t.apply(params, I)
        _, shat, _ = np.linalg.svd(Hhat, full_matrices=False)

        error_s = np.array2string(
            s - shat,
            precision=2,
            separator=", ",
            formatter={"float_kind": lambda x: "%5.2f" % x},
        )
        error_H = f"error_H={jnp.linalg.norm(H-Hhat):6.3f}"
        pbar.set_description(f"{error_s}, {error_H}")

        grads = jax.grad(loss_fn_t.apply)(params, u, v)
        params = jax.tree_util.tree_multimap(update_rule, params, grads)

    I = np.eye(p)
    Hhat = forward_t.apply(params, I)
    loss = jnp.linalg.norm(H - Hhat)
    weights = list(np.array(params[f"lnn/~/linear_{i}"]["w"]) for i in range(3))

    return loss, weights


if __name__ == "__main__":
    cli()

# mypy: ignore-errors

import functools
import itertools
from typing import Sequence
import pprint

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm

from matfact.barcode import Bar
from matfact.analyze_weights import analyze


def data_gen(np_rng, p, H, batch_size):
    while True:
        v = np_rng.normal(size=(batch_size, p)).astype(np.float32)
        yield jnp.array(v), jnp.array(v @ H)


class LNN(nn.Module):
    features: Sequence[int]

    def setup(self) -> None:
        self.layers = [nn.Dense(feat, use_bias=False) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for lyr in self.layers:
          x = lyr(x)
        return x


def create_train_state(rng, p, features, learning_rate, weight_decay):
  """Creates initial `TrainState`."""
  lnn = LNN(features)
  params = lnn.init(rng, jnp.ones((1, p)))['params']
  tx = optax.adamw(learning_rate, weight_decay=weight_decay)
  return train_state.TrainState.create(
      apply_fn=lnn.apply, params=params, tx=tx)


@functools.partial(jax.jit, static_argnums=(0,))
def train_step(features, state, X, y):
    """Train for a single step."""
    def loss_fn(params):
      outputs = LNN(features).apply({'params': params}, X)
      loss = jnp.mean(jnp.sum((outputs - y)**2, axis=1))
      return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def train(ground_truth, p, features, epochs, quiet) -> tuple[float, list[np.ndarray]]:
    I = np.eye(p)

    np_rng = np.random.default_rng(42)
    u = np_rng.normal(size=p)
    H = ground_truth(u)

    batch_size = 256
    input_dataset = data_gen(np_rng, p, H, batch_size)

    s = np.linalg.svd(H, compute_uv=False)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, p, features, 0.001, 0.0001)
    del init_rng

    forward = jax.jit(LNN(features).apply)

    barcode = ''
    for i, (X, y) in enumerate(pbar := tqdm(itertools.islice(input_dataset, epochs), disable=quiet)):
        Hhat = forward({'params': state.params}, I)
        shat = np.linalg.svd(Hhat, compute_uv=False)

        if i % 8 == 0:
            weights = list(np.array(param['kernel']) for param in state.params.values())
            bars = analyze(weights)
            barcode = format_barcode(bars)

        if not quiet:
            error_s = np.array2string(
                s - shat,
                precision=2,
                separator=", ",
                formatter={"float_kind": lambda x: "%5.2f" % x},
            )
            error_H = f"error_H={jnp.linalg.norm(H-Hhat):6.3f}"
            pbar.write("-" * 72)
            pbar.write(f"{error_s}, {error_H}\n{barcode}")

        state = train_step(features, state, X, y)

    Hhat = LNN(features).apply({'params': state.params}, I)
    loss = jnp.linalg.norm(H - Hhat)
    weights = list(np.array(param['kernel']) for param in state.params.values())

    return loss, weights


def format_barcode(bars: list[Bar]) -> str:
    bars_transposed = [[] for _ in bars[0].coefs]
    for i in range(len(bars_transposed)):
        for bar in bars:
            bars_transposed[i].append("%5.1f" % bar.coefs[i])
    barcode = pprint.pformat(bars_transposed)
    return barcode

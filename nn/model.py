from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState


class CNN(nn.Module):
    """A simple CNN model."""

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=10)

    def __call__(self, x):
        for step in self.steps:
            x = step(x)
        return x

    @property
    def steps(self):
        return [self.f1, self.f2, self.f3, self.f4]

    def f1(self, x):
        x = self.conv1(x)
        return x

    def f2(self, x):
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = self.conv2(x)
        return x

    def f3(self, x):
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((*x.shape[:-3], -1))  # flatten
        x = self.dense1(x)
        return x

    def f4(self, x):
        x = nn.relu(x)
        x = self.dense2(x)
        return x


def create_train_state(key, learning_rate, specimen):
    cnn = CNN()
    params = cnn.init(key, specimen)
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


@jax.jit
def train_step(state, image, label):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        logits = state.apply_fn(params, image)
        loss = optax.softmax_cross_entropy(logits, label)
        hit = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(label, axis=-1))
        return loss.sum(), hit

    (loss, hit), grads = loss_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, hit / label.shape[0]


@jax.jit
def test_step(state, image):
    logits = state.apply_fn(state.params, image)
    label = jnp.argmax(logits, axis=-1)
    norm = jnp.sum(jnp.exp(logits), axis=-1)

    return label, norm.sum()

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .linearize import linearize
from .attack import pgd_attack


class CNN(nn.Module):
    """A simple CNN model."""
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.dense1 = nn.Dense(features=256)
        self.dense2 = nn.Dense(features=10)

    def __call__(self, x):
        return self.f4(self.f3(self.f2(self.f1(x))))

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
    return TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx
    )


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
    return state, loss, hit/label.shape[0]


@jax.jit
def test_step(state, image):
    logits = state.apply_fn(state.params, image)
    label = jnp.argmax(logits, axis=-1)

    return label


if __name__ == '__main__':
    learning_rate = 1e-3
    num_epochs = 8
    batch_size = 64

    root = '/mnt/disks/persist/torchvision/datasets'
    train_dataset = MNIST(root, train=True, download=True, transform=np.array)
    test_dataset = MNIST(root, train=False, download=True, transform=np.array)
    specimen = jnp.empty((28, 28, 1))

    key = jax.random.PRNGKey(42)
    state = create_train_state(key, learning_rate, specimen)

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size)
        pbar = tqdm(train_loader)
        for X, y in pbar:
            image = jnp.array(X).reshape((-1, *specimen.shape))/255.
            label = jax.nn.one_hot(jnp.array(y), 10)
            state, loss, accuracy = train_step(state, image, label)
            pbar.set_description(f'{loss.item()=:.2f}, {accuracy.item()=:.2f}')


    # Linearization
    cnn = CNN()

    X, _ = train_dataset[42]
    output = image = jnp.array(X).reshape(specimen.shape)/255.

    # Calculate Jacobian matrices separately
    Js, output = linearize(state, cnn, image)
    J_x = jnp.eye(np.prod(image.shape))
    for J in Js:
        J = J.reshape((J.shape[0], np.prod(J.shape[1:]))).T
        J_x = J @ J_x

    print('Setp-by-step', output)
    eigen_x = jnp.linalg.svd(J_x, compute_uv=False)
    print('Step-by-step', J_x.shape)
    print('Step-by-step', eigen_x)
 
    # Calculate the product of all Jacobian matrices directly
    y, vjp_fun = jax.vjp(cnn.apply, state.params, image)
    J_params, J_x = jax.vmap(vjp_fun)(jnp.eye(10))  # the cotangent vector space is R^10, since there are 10 logits

    print('Streamlined', y)
    eigen_x = jnp.linalg.svd(J_x.reshape((J_x.shape[0], -1)), compute_uv=False)
    print('Streamlined', J_x.shape)
    print('Streamlined', eigen_x)


    # Construct adversial examples with PGD
    exported = set()
    test_loader = DataLoader(test_dataset, 1024)
    for i, (X, y) in enumerate(test_loader):
        image = jnp.array(X).reshape((-1, *specimen.shape))/255.
        label = jnp.array(y)

        target = jax.nn.one_hot((label + 1) % 10, 10)
        adversary = pgd_attack(image, target, state)

        for i in range(label.shape[0]):
            single_label = label[i].item()
            if single_label not in exported:
                exported.add(single_label)

                fig, axes = plt.subplots(1, 2, constrained_layout=True)
                axes[0].imshow(image[i], cmap='gist_gray')
                axes[1].imshow(adversary[i], cmap='gist_gray')
                for ax in axes:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.suptitle(f'Label = {single_label}, Target = {(single_label + 1) % 10}')
                plt.savefig(f'nn/generated/adversarial_{single_label}.png', dpi=200)
                plt.close()

        predicted_orig = test_step(state, image)
        predicted_adv = test_step(state, adversary)

        print(jnp.sum(predicted_orig == label), jnp.sum(predicted_adv == label))

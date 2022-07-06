import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from datasets.load import load_dataset
from tqdm import tqdm


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


def create_train_state(key, learning_rate):
    cnn = CNN()
    params = cnn.init(key, jnp.empty((28, 28, 1)))
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


if __name__ == '__main__':
    learning_rate = 1e-3
    num_epochs = 2
    batch_size = 32

    mnist = load_dataset('mnist', split='train')

    key = jax.random.PRNGKey(42)
    state = create_train_state(key, learning_rate)

    for epoch in range(num_epochs):
        pbar = tqdm(range(0, len(mnist), batch_size))
        for i, batch_id in enumerate(pbar):
            batch = mnist[batch_id:batch_id+batch_size]
            image = jnp.array([np.array(image) for image in batch['image']]).reshape((-1, 28, 28, 1))/255.
            label = jax.nn.one_hot(jnp.array(batch['label']), 10)
            state, loss, accuracy = train_step(state, image, label)
            pbar.set_description(f'{loss.item()=:.2f}, {accuracy.item()=:.2f}')

    cnn = CNN()
    output = X0 = jnp.array([np.array(mnist[42]['image'])]).reshape((28, 28, 1))/255.

    # Calculate Jacobian matrices separately
    J_x = jnp.eye(np.prod(X0.shape))
    for f in (cnn.f1, cnn.f2, cnn.f3, cnn.f4):
        output, J = jax.vmap(
            lambda tangent: jax.jvp(lambda X: cnn.apply(state.params, X, method=f), (output,), (tangent,)),
        )(jnp.eye(np.prod(output.shape)).reshape(-1, *output.shape))
        output = output[0]
        J = J.reshape((J.shape[0], np.prod(J.shape[1:]))).T
        eigen_J = np.linalg.svd(J, compute_uv=False)
        print('Step-by-step (inner)', eigen_J)
        J_x = J @ J_x

    print('Setp-by-step', output)
    eigen_x = jnp.linalg.svd(J_x, compute_uv=False)
    print('Step-by-step', J_x.shape)
    print('Step-by-step', eigen_x)

    # Calculate the product of all Jacobian matrices directly
    y, vjp_fun = jax.vjp(cnn.apply, state.params, X0)
    J_params, J_x = jax.vmap(vjp_fun)(jnp.eye(10))  # the cotangent vector space is R^10, since there are 10 logits
    print('Streamlined', y)
    eigen_x = jnp.linalg.svd(J_x.reshape((J_x.shape[0], -1)), compute_uv=False)
    print('Streamlined', J_x.shape)
    print('Streamlined', eigen_x)


import jax
import jax.numpy as jnp
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import create_train_state, train_step, test_step
from .attack import pgd_attack


if __name__ == "__main__":
    learning_rate = 1e-3
    num_epochs = 8
    batch_size = 64

    root = "/mnt/disks/persist/torchvision/datasets"
    train_dataset = MNIST(root, train=True, download=True, transform=np.array)
    test_dataset = MNIST(root, train=False, download=True, transform=np.array)
    specimen = jnp.empty((28, 28, 1))

    key = jax.random.PRNGKey(42)
    state = create_train_state(key, learning_rate, specimen)

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size)
        pbar = tqdm(train_loader)
        for X, y in pbar:
            image = jnp.array(X).reshape((-1, *specimen.shape)) / 255.0
            label = jax.nn.one_hot(jnp.array(y), 10)
            state, loss, accuracy = train_step(state, image, label)
            pbar.set_description(f"{loss.item()=:.2f}, {accuracy.item()=:.2f}")

    # Construct adversial examples with PGD for the test set
    exported = set()
    test_loader = DataLoader(test_dataset, 1024)

    total_hits_orig, total_hits_adv = 0, 0
    total_norm_orig, total_norm_adv = 0, 0
    for i, (X, y) in enumerate(test_loader):
        image = jnp.array(X).reshape((-1, *specimen.shape)) / 255.0
        label = jnp.array(y)

        target = jax.nn.one_hot((label + 1) % 10, 10)
        adversary = pgd_attack(image, target, state)

        for i in range(label.shape[0]):
            single_label = label[i].item()
            if single_label not in exported:
                exported.add(single_label)

                fig, axes = plt.subplots(1, 2, constrained_layout=True)
                axes[0].imshow(image[i], cmap="gist_gray")
                axes[1].imshow(adversary[i], cmap="gist_gray")
                for ax in axes:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.suptitle(
                    f"Label = {single_label}, Target = {(single_label + 1) % 10}"
                )
                plt.savefig(f"nn/generated/adversarial_{single_label}.png", dpi=200)
                plt.close()

        predicted_orig, norm_orig = test_step(state, image)
        predicted_adv, norm_adv = test_step(state, adversary)

        total_hits_orig += jnp.sum(predicted_orig == label)
        total_hits_adv += jnp.sum(predicted_adv == label)

        total_norm_orig += norm_orig
        total_norm_adv += norm_adv

    total = len(test_dataset)
    print(
        f"Test accuracy: {total_hits_orig/total*100:.2f}% -> {total_hits_adv/total*100:.2f}%"
    )
    print(f"Test confidence: {total_norm_orig/total:.2f} -> {total_norm_adv/total:.2f}")

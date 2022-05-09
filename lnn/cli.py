# mypy: ignore-errors

import click

from lnn.train import train
from lnn.ground_truth import ground_truths


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
@click.option("--p", type=int, help="Dimension of H.")
@click.option(
    "--features",
    type=int,
    multiple=True,
    help="Number of output features in each hidden layer.",
)
@click.option("--epochs", type=int, help="Number of training epochs.")
@click.option("--quiet", type=bool, default=False, help="Suppress output.")
def cli(ground_truth, p, features, epochs, quiet):
    train(ground_truth, p, features, epochs, quiet)


if __name__ == "__main__":
    cli()

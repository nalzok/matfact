# matfact
[![Test Status](https://github.com/nalzok/matfact/actions/workflows/pytest.yml/badge.svg)](https://github.com/nalzok/matfact/actions/workflows/pytest.yml)
[![Lint Status](https://github.com/nalzok/matfact/actions/workflows/black.yml/badge.svg)](https://github.com/nalzok/matfact/actions/workflows/black.yml)

## Roadmap

TODO:
+ [ ] Matrix Factorization
    + [X] Debug issues above
    + [X] Rational arithmetic (try Python libaries first)
    + [X] Shape commutation algorithm (Prop 7)
    + [ ] Quiver algorithm (Alg 5)
        + [X] One direction
        + [ ] The other direction
+ [ ] Poke around
    + [X] Track coefficients of bases
    + [X] Train a linear neural network with JAX+Haiku
    + [X] Extract barcode from the LNN
    + [ ] Track condition number of Bar within each epoch
        + [ ] Train longer to see if condition number drops
    + [ ] Weight shrinkage / dropout
    + [X] Rank one transformation instead of Householder
        + Doesn't seem to work; still gives full-length bars
    + [X] Scale H to have different singular values and use narrower/wider hidden layers
        + Hhat appears to be the closest unitary/orthogonal matrix of H
        + Wider hidden layers leads to faster convergence
        + See for `lnn/RESULTS.md` for simulation results
    + [ ] Determinant of Hhat
    + [ ] Run L(E)UP on H and check U
        + [ ] Maybe on Hhat as well
    + [X] Rename Barcode to Bar


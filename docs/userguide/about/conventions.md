(conventions)=

# Conventions

This page documents the project-wide sign conventions used across all
interaction modules (electrostatics, dispersion, Lennard-Jones).

## Virial

All modules return the virial tensor defined as the negative strain derivative
of the energy:

$$W_{ab} = -\frac{\partial E}{\partial \varepsilon_{ab}}$$

where $\varepsilon$ is the symmetric infinitesimal strain tensor.

For pairwise real-space interactions this is equivalent to:

$$W = -\sum_{i < j} \mathbf{r}_{ij} \otimes \mathbf{F}_{ij}$$

where $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ and $\mathbf{F}_{ij}$
is the force on atom $i$ due to atom $j$.
Individual kernel implementations may use the reversed separation vector or the
reaction force internally, but returned virials always follow this convention.

## Stress

The tensile-positive Cauchy stress is obtained from the virial as:

$$\sigma = -\frac{W}{V}$$

where $V = |\det(\mathbf{C})|$ is the cell volume.

```{note}
Some molecular-dynamics codes use the opposite (compression-positive or
"pressure") convention $\sigma = W / V$. When comparing against external
references, check which convention they follow.
```

## Separation Vector

The canonical separation vector points from atom $i$ to atom $j$:

$$\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$$

Individual kernel implementations may use either direction internally, but the
returned virial always follows the convention above.

# Spectral estimate of entropy production rates via entropy production factor
This repository contains code written in Python to calculate entropy production rates from times series data of random variables and fields. The paper describing the technique described here can be found [here](https://arxiv.org/abs/1911.10696).

### Theory
We solve for the entropy production exhibited by a time series of $`N \geq 2`$ variables over a time $`T`$, $`\mathbf{x}(t)`$, using the information theoretic measure of entropy production introduced in [Kawai, Parrondo, and Van den Broeck, PRL 2007](https://link.aps.org/doi/10.1103/PhysRevLett.98.080602),

```math
\langle \dot{S} \rangle = \lim_{T \to \infty} \dfrac{1}{T} D_{KL}(\mathcal{P}[\mathbf{x}(t)] || \mathcal{P}[\widetilde{\mathbf{x}}(t)]
```

where $`D_{KL}`$ is the Kullback-Leibler divergence, or relative entropy, between the probability functional of observing a forward path, $`\mathcal{P}[\mathbf{x}(t)]`$, and the probability functional of observing its reverse path, $`\mathcal{P}[\widetilde{\mathbf{x}}(t)]`$. We assume $`\mathcal{P}[\mathbf{x}(t)]`$ to be Gaussian,

```math
\mathcal{P}[\mathbf{x}(\omega)] = \dfrac{1}{Z} \exp \left( -\dfrac{1}{2} \int \dfrac{d \omega}{2 \pi} \mathbf{x}^\dagger C^{-1} \mathbf{x} \right)
```

where $`C_{ij}(\omega) = \langle x_i(\omega) x_j(-\omega) \rangle`$ is the frequency space covariance matrix for the variables $`x_i(t)`$ and $`Z = \exp \left(  \frac{T}{2} \int \frac{d \omega}{2\pi} \ \ln \left[ \det C (\omega) \right]  \right)`$ is a normalization constant. The same is done for the reverse path. Solving for $`D_{KL}`$ and taking the relevant limit, the entropy production rate is given by

```math
\dot{S} =\frac{1}{2} \int \frac{d \omega}{2 \pi} \left[ \ln \left(\frac{\det C(-\omega)}{\det C(\omega)} \right) + \left(C^{-1} (-\omega) - C^{-1}(\omega) \right)_{ij} C^{ji}(\omega) \right]
```

This expression exists not only for random variables $`\mathbf{x}(t)`$, but also for random fields, $`\boldsymbol{\phi}(\mathbf{r}, t)`$, where $`\mathbf{r} \in \mathbb{R}^d`$. In this case, the expressions given above are virtually unchanged, but have additional integrals over the spatial wavevectors, $`\mathbf{k}`$.

### Code
The code to calculate the entropy production rate is written as a module called `freqent` (i.e. **freq**uency **ent**ropy) for easy use and modularity. After cloning the repository, the module is easily installed using `pip`:

```bash
cd /path/to/this/repo
pip install -e .
```
There are 2 `yml` files in the repository that can be used to create a virtual environment in which to run the module. One has requirments for Macs and the other for Linux. The code has not been tested on Windows machines. To create the virtual environment using [`conda`](https://docs.conda.io/en/latest/):

```bash
conda env create -f epf_paper_osx.yml
```

There are two submodules, `freqent.freqent` for use with random variables and `freqent.freqentn` for random fields (similar to `numpy.fft.fft` vs. `numpy.fft.fftn`).
Once installed, the methods can be called from within a script, Jupyter notebook, or iPython terminal by importing the relevant module as you would any other one:

```python
import freqent.freqent as fe
import freqent.freqentn as fen
```

The main functions to use are `fe.entropy()` and `fen.entropy()`. See their documentation in `freqent/freqent.py` and `freqent/freqentn.py` respectively.

There are several simulations present in the `freqent/tests/` folder that output data ready for input into the relevant `entropy()` functions. Examples on how to run each simulation is in a `README` file in each of simulation's folder.

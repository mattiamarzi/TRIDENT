# Mathematical notes

TRIDENT implements statistically validated projections of bipartite data.
The core objects are:

- a binary biadjacency matrix $B = (b_{i\alpha})$ with $b_{i\alpha} \in \{0,1\}$,
- pair co-occurrences
  $V_{ij} = \sum_{\alpha} b_{i\alpha} b_{j\alpha}$,
- triplet co-occurrences
  $V_{ijk} = \sum_{\alpha} b_{i\alpha} b_{j\alpha} b_{k\alpha}$.

Statistical validation is performed by comparing $V_{ij}$ and $V_{ijk}$ to their
distribution under maximum-entropy null models.

## BiCM link probabilities

The Bipartite Configuration Model (BiCM) constrains, in expectation, the degrees of both layers:

$$
    \mathbb{E}[k_i] = k_i^*, \qquad k_i = \sum_{\alpha} b_{i\alpha}, \\
    \mathbb{E}[h_\alpha] = h_\alpha^*, \qquad h_\alpha = \sum_i b_{i\alpha}.
$$

The model factorizes over edges and yields

$$
    p_{i\alpha} \equiv \mathbb{P}(b_{i\alpha}=1)
    = \frac{x_i y_\alpha}{1 + x_i y_\alpha}.
$$

The parameters $x_i$ and $y_\alpha$ are fitted by matching the constraints.

## BiPLOM link probabilities

The Bipartite Partial Local Overlap Model (BiPLOM) augments the BiCM constraints by
controlling, for each node $i$ in the bottom layer, its aggregate tendency to
co-activate with the rest of the layer.

Rather than constraining the strict local overlap
$V_i = \sum_{j \ne i} \sum_{\alpha} b_{i\alpha} b_{j\alpha}$,
which leads to degeneracies and prevents convergence at finite size,
BiPLOM enforces a softened, mean-field constraint on the quantity

$$
    \tilde V_i
    \equiv \sum_{j} \sum_{\alpha} b_{i\alpha} b_{j\alpha}
    = \sum_{\alpha} b_{i\alpha} h_\alpha,
$$

and requires

$$
    \left\langle \tilde V_i \right\rangle = \tilde V_i^* \qquad \forall i.
$$

This approximation amounts to including the diagonal contribution ($j = i$) in
the overlap term and avoids the collapse of degree fluctuations that would occur
when enforcing the strict $j \ne i$ constraint.

Under a mean-field approximation, the effective link probabilities can be written as

$$
    p_{i\alpha} = \frac{\exp\left(-\alpha_i - \beta_\alpha + \delta_i h_\alpha^*\right)}{1 + \exp\left(-\alpha_i - \beta_\alpha + \delta_i h_\alpha^*\right)}.
$$

This parameterization ensures the simultaneous reproduction of the degree
constraints and of the aggregate overlap $\tilde V_i$ in expectation.

A detailed discussion of the origin of the $\tilde V_i$ constraint, its relation to the strict local overlap $V_i$, and the associated convergence properties of the solver is provided in the companion manuscript introducing BiPLOM.

## Poisson-binomial distributions for motifs

### Pairs

Under independent-link models, each summand $b_{i\alpha} b_{j\alpha}$ is a Bernoulli variable with success probability

$$
    q_\alpha^{(ij)} = p_{i\alpha} p_{j\alpha}.
$$

Therefore $V_{ij}$ is Poisson-binomial:

$$
    V_{ij} = \sum_{\alpha=1}^M X_\alpha, \qquad X_\alpha \sim \mathrm{Bernoulli}\left(q_\alpha^{(ij)}\right),
$$

and TRIDENT evaluates the right-tail p-value $\mathbb{P}(V_{ij} \ge V_{ij}^*)$.

### Triplets

Similarly,

$$
    q_\alpha^{(ijk)} = p_{i\alpha} p_{j\alpha} p_{k\alpha},
$$

and $V_{ijk}$ is Poisson-binomial with heterogeneous probabilities $q_\alpha^{(ijk)}$.

## Approximations

When an exact Poisson-binomial evaluation is unnecessary or too expensive, TRIDENT implements common approximations:

- **Poisson**: $V \approx \mathrm{Poisson}(\mu)$ with $\mu = \sum_\alpha q_\alpha$.
- **Normal (CLT)**: $V \approx \mathcal{N}(\mu, \sigma^2)$ with
  $\sigma^2 = \sum_\alpha q_\alpha (1 - q_\alpha)$, using continuity correction.
- **Refined normal**: a skewness-corrected Gaussian approximation.

For the exact Poisson-binomial probability mass function, TRIDENT uses an FFT-based method.

## Multiple-testing correction

Both pairwise and triadic validation in TRIDENT involve a large number of simultaneous hypothesis tests:
$\binom{N}{2}$ tests for pairs and $\binom{N}{3}$ tests for triplets. To control false positives, raw p-values
must therefore be corrected for multiple comparisons.

TRIDENT implements the following correction procedures:

- **Bonferroni correction**
  Controls the family-wise error rate (FWER) by rescaling the significance level as $\alpha / m$, where $m$ is
  the total number of tests. This approach is exact but extremely conservative when $m$ is large, often resulting
  in a substantial loss of statistical power.

- **False Discovery Rate (FDR)**
  Implements the Benjamini–Hochberg procedure, controlling the expected fraction of false positives among
  rejected hypotheses. FDR provides a more favorable balance between sensitivity and specificity in large-scale
  screening problems and is therefore recommended as the default choice in most applications.

Corrections are applied independently for pairs and triplets, as they correspond to different null hypotheses
(BiCM and BiPLOM, respectively) and involve different numbers of statistical tests.

## Related references

The BiCM and the statistical validation of bipartite projections are described in:

- F. Saracco, R. Di Clemente, A. Gabrielli, T. Squartini, "Randomizing bipartite networks: the case of the World Trade Web", Scientific Reports 5, 10595 (2015).
- F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, T. Squartini, "Inferring monopartite projections of bipartite networks: an entropy-based approach", New Journal of Physics 19, 053022 (2017).

Efficient likelihood maximization strategies for ERGMs with local constraints are discussed in:

- N. Vallarano, M. Bruno, E. Marchese, G. Trapani, F. Saracco, T. Squartini, G. Cimini, M. Zanon, "Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints", Scientific Reports (2021).

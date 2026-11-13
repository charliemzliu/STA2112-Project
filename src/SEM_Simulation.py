"""Simulate observational and do-interventional linear Gaussian DAG SEMs.

A total of p structural variables Z_0, ..., Z_{p-1} are generated via
    Z_i = mean[i] + sum_j Coeff[i, j] * Z_j + epsilon_i,
with mutually independent epsilon_i ~ N(0, Variance[i]).
"""

import numpy as np

try:  # When imported as src.Linear_Gaussian_SEM
    from .InterventionData import InterventionData
except ImportError:  # When the files are on the same Python import path
    from InterventionData import InterventionData


def _positive_integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _prepare_sem(p, Coeff, mean, Variance, target):
    """Validate inputs and find a topological ordering of the DAG."""
    p = _positive_integer(p, "p")
    if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)):
        raise TypeError("target must be a 0-based integer index.")
    target = int(target)
    if not 0 <= target < p:
        raise ValueError("target must be between 0 and p - 1.")

    coeff = np.asarray(Coeff, dtype=float)
    means = np.asarray(mean, dtype=float)
    variances = np.asarray(Variance, dtype=float)

    if coeff.shape != (p, p):
        raise ValueError(f"Coeff must have shape ({p}, {p}).")
    if means.shape != (p,):
        raise ValueError(f"mean must have shape ({p},).")
    if variances.ndim == 0:
        variances = np.full(p, float(variances))
    elif variances.shape != (p,):
        raise ValueError(f"Variance must be a scalar or have shape ({p},).")

    if not (np.isfinite(coeff).all() and np.isfinite(means).all()
            and np.isfinite(variances).all()):
        raise ValueError("Coeff, mean and Variance must contain finite values.")
    if np.any(variances < 0):
        raise ValueError("Noise variances must be nonnegative.")
    if np.any(np.diag(coeff) != 0):
        raise ValueError("Coeff must have a zero diagonal (no self-loops).")

    # Coeff[i,j] != 0 represents an edge j -> i.
    indegree = np.count_nonzero(coeff, axis=1).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    order = []
    while ready:
        j = ready.pop(0)
        order.append(int(j))
        for child in np.flatnonzero(coeff[:, j] != 0):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))

    if len(order) != p:
        raise ValueError("Coeff contains a directed cycle; the SEM must be a DAG.")
    return coeff, means, np.sqrt(variances), target, order


def _simulate(n, coeff, means, noise_sds, order, do_values=None):
    """Generate n iid samples of all p structural variables."""
    p = len(means)
    z = np.empty((n, p), dtype=float)
    do_values = {} if do_values is None else do_values

    for j in order:
        if j in do_values:
            # Perfect intervention: delete incoming edges and replace the noise.
            z[:, j] = do_values[j]
        else:
            parents = np.flatnonzero(coeff[j, :] != 0)
            z[:, j] = means[j] + np.random.normal(0, noise_sds[j], size=n)
            if len(parents):
                z[:, j] += z[:, parents] @ coeff[j, parents]
    return z


def _split_target(z, target):
    """Return (predictors, target) with target excluded from predictors."""
    return np.delete(z, target, axis=1), z[:, target].copy()


def _causal_indices(coeff, target):
    """Return direct parents in *predictor-column* indexing."""
    predictor_nodes = [j for j in range(coeff.shape[0]) if j != target]
    return tuple(k for k, j in enumerate(predictor_nodes) if coeff[target, j] != 0)


def Linear_Gussian_SEM(n, p, Coeff, mean, Variance, target):
    """Return an observational InterventionData object (environment e=1).

    `p` is the total number of SEM variables, including the target.
    Coeff[i,j] is the effect of variable j on variable i.
    `mean` contains structural intercepts, not necessarily marginal means.
    `Variance` is a scalar or p-vector of noise variances.
    """
    n = _positive_integer(n, "n")
    coeff, means, noise_sds, target, order = _prepare_sem(
        p, Coeff, mean, Variance, target
    )
    z = _simulate(n, coeff, means, noise_sds, order)
    x, y = _split_target(z, target)
    return InterventionData(
        X=x,
        Y=y,
        environments=np.ones(n, dtype=int),
        Causals=_causal_indices(coeff, target),
    )


def Linear_Gussian_DO(size, p, Coeff, mean, Variance, target, intervention):
    """Return observational and perfect-do interventional environments.

    
    size: lengths for all environments, including observational e=1.
        Accepts a vector with length E = 1 + len(intervention), or an
        (E, 1) column array. size[e-1] is the sample size of environment e.
        A positive integer scalar is also supported for backward compatibility
        and assigns the same sample size to every environment.
    p: Number of variables in this model, including the target variable.
    Coeff: An (p,p) array reprensting the linear gussian coefficient matrix.
    mean: A (p,1) array crspding to all intercepts when generating this model.
    Variance: A (p,1) array or a scalar of non-negative floats crspding to variance
    of the noise in the generation. If Variance is a scalar, then all noise variance 
    are set to this scalar.
    target: The target variable's index, notice the index start at 0.
    intervention: a list of (indices, values) pairs; each pair defines a
        separate environment, e.g. [([0], [4.0]), ([0,2], [1.0,-3.0])].
    Do-interventions on the target are prohibited.
    """
    coeff, means, noise_sds, target, order = _prepare_sem(
        p, Coeff, mean, Variance, target
    )

    # Validate all interventions before starting any random sampling.
    interventions = []
    for env, specification in enumerate(intervention, start=2):
        if not isinstance(specification, (list, tuple)) or len(specification) != 2:
            raise ValueError(f"Intervention {env} must be (indices, values).")

        indices, values = specification
        indices = np.asarray(indices)
        values = np.asarray(values, dtype=float)

        if indices.ndim != 1 or values.ndim != 1 or len(indices) != len(values):
            raise ValueError(f"Intervention {env} requires two equal-length 1D arrays.")

        if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
               for j in indices):
            raise TypeError(f"Intervention {env} indices must be integers.")

        if len(set(indices.tolist())) != len(indices):
            raise ValueError(f"Intervention {env} repeats an index.")

        if any(j < 0 or j >= len(means) for j in indices):
            raise ValueError(f"Intervention {env} indices must be between 0 and p-1.")

        if target in indices:
            raise ValueError(f"Intervention {env} cannot intervene on target {target}.")

        if not np.isfinite(values).all():
            raise ValueError(f"Intervention {env} values must be finite.")

        interventions.append(dict(zip(map(int, indices), map(float, values))))

    # There are E environments: 1 observational plus one per do-intervention.
    E = len(interventions) + 1

    if isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)):
        # Preserve the original scalar-size API.
        sizes = [_positive_integer(size, "size")] * E
    else:
        sizes_array = np.asarray(size, dtype=object)

        if sizes_array.ndim == 2 and sizes_array.shape[1] == 1:
            sizes_array = sizes_array[:, 0]

        if sizes_array.ndim != 1 or len(sizes_array) != E:
            raise ValueError(f"size must contain {E} entries (one per environment).")

        sizes = [
            _positive_integer(count, f"size[{e - 1}]")
            for e, count in enumerate(sizes_array, start=1)
        ]

    # Generate observational data (e = 1).
    z_obs = _simulate(sizes[0], coeff, means, noise_sds, order)
    blocks = {1: _split_target(z_obs, target)}

    # Generate each interventional environment (e = 2, 3, ...).
    for env, do_values in enumerate(interventions, start=2):
        z_int = _simulate(
            sizes[env - 1], coeff, means, noise_sds, order, do_values
        )
        blocks[env] = _split_target(z_int, target)

    return InterventionData.from_environment_blocks(
        blocks, Causals=_causal_indices(coeff, target)
    )



"""                       Work in progress                  """

def Linear_Gussian_Noise()


def 








"""Simulate observational and do-interventional linear Gaussian DAG SEMs.

A total of p structural variables Z_0, ..., Z_{p-1} are generated via
    Z_i = mean[i] + sum_j Coeff[i, j] * Z_j + epsilon_i,
with mutually independent epsilon_i ~ N(0, Variance[i]). Hidden nodes are
simulated but not observed, and cannot be directly intervened on.
"""

import numpy as np

try:  # When imported as src.SEM_Simulation
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


def _validate_hidden(p, target, hidden):
    """Validate hidden indices in the full SEM."""
    if hidden is None:
        return ()
    if isinstance(hidden, (str, bytes)):
        raise TypeError("hidden must be an iterable of 0-based integer indices.")
    try:
        nodes = tuple(hidden)
    except TypeError as exc:
        raise TypeError("hidden must be an iterable of 0-based integer indices.") from exc
    if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
           for j in nodes):
        raise TypeError("hidden must contain integer SEM indices.")
    nodes = tuple(int(j) for j in nodes)
    if len(set(nodes)) != len(nodes):
        raise ValueError("hidden cannot contain duplicate indices.")
    if any(j < 0 or j >= p for j in nodes):
        raise ValueError("hidden indices must be between 0 and p - 1.")
    if target in nodes:
        raise ValueError("The target variable cannot be hidden.")
    return nodes


def _predictor_nodes(p, target, hidden=()):
    """Map predictor-column indices to full SEM indices, in SEM order."""
    excluded = set(hidden) | {target}
    return [j for j in range(p) if j not in excluded]


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


def _split_target(z, target, hidden=()):
    """Return observed predictors and target after full SEM simulation."""
    return z[:, _predictor_nodes(z.shape[1], target, hidden)], z[:, target].copy()


def _causal_indices(coeff, target, hidden=()):
    """Return observed direct parents in predictor-column indexing."""
    predictor_nodes = _predictor_nodes(coeff.shape[0], target, hidden)
    return tuple(k for k, j in enumerate(predictor_nodes) if coeff[target, j] != 0)


def Linear_Gaussian_SEM(n, p, Coeff, mean, Variance, target, hidden=None):
    """Return an observational InterventionData object (environment e=1).

    p is the total number of SEM variables, including target and hidden nodes.
    Coeff[i,j] is the effect of variable j on variable i.
    mean contains structural intercepts, not necessarily marginal means.
    Variance is a scalar or p-vector of noise variances.
    hidden is an optional iterable of 0-based full SEM indices. Hidden nodes
    are generated normally but excluded from X. Causals contains only observed
    direct parents in predictor-column indexing.
    """
    n = _positive_integer(n, "n")
    coeff, means, noise_sds, target, order = _prepare_sem(
        p, Coeff, mean, Variance, target
    )
    hidden = _validate_hidden(len(means), target, hidden)
    z = _simulate(n, coeff, means, noise_sds, order)
    x, y = _split_target(z, target, hidden)
    return InterventionData(
        X=x,
        Y=y,
        environments=np.ones(n, dtype=int),
        Causals=_causal_indices(coeff, target, hidden),
    )


def Linear_Gaussian_DO(size, p, Coeff, mean, Variance, target, intervention, hidden=None):
    """Return observational and perfect-do interventional environments.

    size: lengths for all environments, including observational e=1.
        Accepts a vector with length E = 1 + len(intervention), or an
        (E, 1) column array. size[e-1] is the sample size of environment e.
        A positive integer scalar is also supported for backward compatibility
        and assigns the same sample size to every environment.
    p: Total number of variables, including target and hidden nodes.
    Coeff: A (p, p) coefficient matrix; Coeff[i,j] is the effect of j on i.
    mean: A (p,) vector of structural intercepts.
    Variance: A (p,) vector or scalar of nonnegative noise variances.
    target: The target's 0-based index in the full SEM.
    intervention: a list of (indices, values) pairs; each pair defines a
        separate environment, e.g. [([0], [4.0]), ([0,2], [1.0,-3.0])].
    hidden: Optional iterable of 0-based full SEM indices of hidden variables.
        Hidden nodes may occur anywhere in the DAG and are simulated normally.
    Intervention indices refer to the full SEM, not columns of returned X.
    Do-interventions on the target and hidden variables are prohibited.
    X contains observed predictors in original SEM order, and Causals contains
    only observed direct parents in predictor-column indexing.
    """
    coeff, means, noise_sds, target, order = _prepare_sem(
        p, Coeff, mean, Variance, target
    )
    hidden = _validate_hidden(len(means), target, hidden)

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

        hidden_targets = set(indices.tolist()).intersection(hidden)
        if hidden_targets:
            raise ValueError(f"Intervention {env} cannot intervene on hidden nodes "
                             f"{sorted(hidden_targets)}.")

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
    blocks = {1: _split_target(z_obs, target, hidden)}

    # Generate each interventional environment (e = 2, 3, ...).
    for env, do_values in enumerate(interventions, start=2):
        z_int = _simulate(
            sizes[env - 1], coeff, means, noise_sds, order, do_values
        )
        blocks[env] = _split_target(z_int, target, hidden)

    return InterventionData.from_environment_blocks(
        blocks, Causals=_causal_indices(coeff, target, hidden)
    )



"""                       Work in progress                  """

#def Linear_Gaussian_Noise()


#def

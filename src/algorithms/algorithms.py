"""
Search methods for fitting the smallest enclosing circle to a point set

All methods search the same three parameters (cx, cy, r) and use the same
batched cost function, so the results can be compared fairly
Every method returns (best, history), where history holds the best cost
found so far after each iteration (generation for the GA)

One iteration evaluates this many circles:
    hill climbing:        restarts
    simulated annealing:  chains
    genetic algorithm:    pop_size
"""

import numpy as np
from model.base import Circle, CostBatch


def diagonal(pts):
    """
    Measure the size of the data, used to make step sizes independent of scale

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points

    Returns:
    float: The length of the diagonal of the bounding box
    """
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def random_start(pts, rng):
    """
    Create a random circle inside the bounding box of the data

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    rng (np.random.Generator): The random number generator

    Returns:
    Circle: A circle with a random center and a radius between 10 and 50
        percent of the bounding box diagonal
    """
    # the bound of the data
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # the centers 
    cx, cy = rng.uniform(lo, hi)
    # radius
    r = rng.uniform(0.1, 0.5) * np.linalg.norm(hi - lo)
    return Circle(cx, cy, r)


def finish(pts, x):
    """
    Turn a parameter vector into the final circle, tightened to the farthest point

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    x (np.ndarray): The parameters (cx, cy, r) of the best solution

    Returns:
    Circle: The circle with the given center that just encloses every point
    """
    return Circle(x[0], x[1], x[2]).tighten(pts)


def hill_climbing(pts, max_iter, step0, restarts, patience, rng, chunk=256):
    """
    Hill climbing with random restarts, all restarts run together as one numpy batch

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    max_iter (int): The maximum number of iterations
    step0 (float): The initial standard deviation of the random steps, given
        as a fraction of the bounding box diagonal (for example 0.1)
    restarts (int): The number of random restarts
    patience (int): The number of failed iterations before the step is halved
    rng (np.random.Generator): The random number generator
    chunk (int): The number of iterations whose random noise is drawn at once,
        a larger value is faster but uses more memory

    Returns:
    best (Circle): The best circle found, tightened to the farthest point
    history (np.ndarray): The best cost over all restarts after each iteration
    """
    cost_of = CostBatch(pts, restarts)

    # one row per restart: cx, cy, r
    X = np.array([random_start(pts, rng).params() for _ in range(restarts)])
    cost = cost_of(X).copy()

    step = np.full(restarts, step0 * diagonal(pts))
    fails = np.zeros(restarts, dtype=np.int64)
    history = np.empty(max_iter)
    cand = np.empty_like(X)

    # one iteration is one step for every restart
    for start in range(0, max_iter, chunk):
        stop = min(start + chunk, max_iter)
        noise = rng.standard_normal((stop - start, restarts, 3))

        for i in range(start, stop):
            np.multiply(noise[i - start], step[:, None], out=cand)
            cand += X
            # the radius stays non negative
            np.abs(cand[:, 2], out=cand[:, 2])
            cand_cost = cost_of(cand)

            better = cand_cost < cost
            X[better] = cand[better]
            cost[better] = cand_cost[better]

            fails += 1
            fails[better] = 0
            stuck = fails >= patience
            # restarts that are stuck get a finer step
            step[stuck] *= 0.5
            fails[stuck] = 0

            history[i] = cost.min()

    # the best circle is the one with the lowest cost over all restarts
    k = int(cost.argmin())
    return finish(pts, X[k]), history


def simulated_annealing(pts, max_iter, T0, alpha, step0, chains, rng,
                        T_min=1e-6):
    """
    Simulated annealing with several independent chains run as one numpy batch

    The step size shrinks together with the temperature, and worse solutions
    are accepted with probability exp(-delta / T)

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    max_iter (int): The number of iterations
    T0 (float): The starting temperature, given as a fraction of the
        bounding box diagonal (the cost is measured in length units)
    alpha (float): The geometric cooling factor, usually between 0.995 and 0.999
    step0 (float): The step size at the starting temperature, given as a
        fraction of the bounding box diagonal
    chains (int): The number of independent chains
    rng (np.random.Generator): The random number generator
    T_min (float): The lowest temperature, given as a fraction of the
        bounding box diagonal

    Returns:
    best (Circle): The best circle found, tightened to the farthest point
    history (np.ndarray): The best cost over all chains after each iteration
    """
    cost_of = CostBatch(pts, chains)
    diag = diagonal(pts)

    # setting up staring arrays for the chains, one row per chain: cx, cy, r
    X = np.array([random_start(pts, rng).params() for _ in range(chains)])
    cost = cost_of(X).copy()
    best_X = X.copy()
    best_cost = cost.copy()

    # starting parameters for temperatures
    T_start = T0 * diag
    T_low = T_min * diag
    T = T_start
    history = np.empty(max_iter)

    # one iteration is one step for every chain
    for i in range(max_iter):
        step = step0 * diag * (T / T_start)
        cand = X + rng.standard_normal((chains, 3)) * step
        cand[:, 2] = np.abs(cand[:, 2])
        cand_cost = cost_of(cand)

        # better moves always pass, worse ones pass with a small chance
        delta = cand_cost - cost
        chance = np.exp(-np.maximum(delta, 0) / T)
        accept = (delta < 0) | (rng.random(chains) < chance)
        X[accept] = cand[accept]
        cost[accept] = cand_cost[accept]

        improved = cost < best_cost
        best_X[improved] = X[improved]
        best_cost[improved] = cost[improved]

        # temperature is lowred, but never below T_low
        T = max(T * alpha, T_low)
        history[i] = best_cost.min()

    # the best circle is the one with the lowest cost over all chains
    k = int(best_cost.argmin())
    return finish(pts, best_X[k]), history


def tournament(cost, n, k, rng):
    """
    Pick n parents with tournament selection

    Parameters:
    cost (np.ndarray): The cost of every individual, shape (pop_size,)
    n (int): The number of parents to pick
    k (int): The number of random contestants in every tournament
    rng (np.random.Generator): The random number generator

    Returns:
    np.ndarray: The indices of the n winners, shape (n,), lower cost wins
    """
    idx = rng.integers(0, len(cost), (n, k))
    winners = np.argmin(cost[idx], axis=1)
    return idx[np.arange(n), winners]


def genetic_algorithm(pts, pop_size, max_gen, pc, pm, sigma0, n_elite,
                      k_tourn, rng):
    """
    Genetic algorithm with tournament selection, arithmetic crossover,
    Gaussian mutation and elitism, the whole population is evaluated as one batch

    Parameters:
    pts (np.ndarray): An array of shape (N, 2) with the points
    pop_size (int): The number of individuals
    max_gen (int): The number of generations
    pc (float): The crossover probability
    pm (float): The mutation probability
    sigma0 (float): The mutation strength in the first generation, given as a
        fraction of the bounding box diagonal, it shrinks linearly later
    n_elite (int): The number of best individuals copied unchanged
    k_tourn (int): The tournament size
    rng (np.random.Generator): The random number generator

    Returns:
    best (Circle): The best circle found, tightened to the farthest point
    history (np.ndarray): The best cost in the population after each generation
    """

    # the cost function for the whole population, evaluated in one batch
    cost_of = CostBatch(pts, pop_size)
    diag = diagonal(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    n_child = pop_size - n_elite

    # the population is a 2D array with one row per individual: cx, cy, r
    pop = np.array([random_start(pts, rng).params() for _ in range(pop_size)])
    cost = cost_of(pop).copy()
    history = np.empty(max_gen)

    # one generation is one step for every individual
    for gen in range(max_gen):
        # the mutation gets weaker, but never reaches zero
        sigma = sigma0 * diag * max(1 - gen / max_gen, 0.01)

        order = np.argsort(cost)
        new_pop = np.empty_like(pop)
        new_pop[:n_elite] = pop[order[:n_elite]]

        # pick parents and create children
        p1 = pop[tournament(cost, n_child, k_tourn, rng)]
        p2 = pop[tournament(cost, n_child, k_tourn, rng)]

        # arithmetic crossover, the others are plain copies of the first parent
        a = rng.random((n_child, 1))
        cross = rng.random(n_child) < pc
        child = np.where(cross[:, None], a * p1 + (1 - a) * p2, p1)

        # mutate the children, the radius is kept non negative
        mutate = rng.random(n_child) < pm
        child += mutate[:, None] * rng.standard_normal((n_child, 3)) * sigma

        # keep the center in the data box and the radius in a sane range
        child[:, 0] = np.clip(child[:, 0], lo[0], hi[0])
        child[:, 1] = np.clip(child[:, 1], lo[1], hi[1])
        child[:, 2] = np.clip(np.abs(child[:, 2]), 0, diag)

        # the new population is the elite plus the children
        new_pop[n_elite:] = child
        pop = new_pop
        cost = cost_of(pop).copy()
        history[gen] = cost.min()

    # the best circle is the one with the lowest cost over all generations
    k = int(cost.argmin())
    return finish(pts, pop[k]), history
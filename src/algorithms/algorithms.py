import numpy as np
from model.base import Circle, CostBatch


def random_start(pts, rng):
    """Random circle inside the bounding box of the data"""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = rng.uniform(lo, hi)
    r = rng.uniform(0.1, 0.5) * np.linalg.norm(hi - lo)
    return Circle(cx, cy, r)


def hill_climbing(pts, max_iter, step0, restarts, patience, rng, chunk=256):
    """
    Hill climbing, all restarts run together as one numpy batch
 
    
    Parameters:
    pts: array of shape (N, 2)
    max_iter: maximum number of iterations
    step0: initial step size for random noise
    restarts: number of random restarts
    patience: number of iterations to wait for improvement before reducing step size
    rng: numpy random generator
    chunk: number of iterations to process at once, controls how many iterations of random noise are
 
    Returns:
    best (Circle): the best circle found
    history (np.ndarray): best cost over all restarts after each iteration
    """
    # allocate buffers for cost evaluation
    cost_of = CostBatch(pts, restarts)
 
    # one row per restart: cx, cy, r
    X = np.array([random_start(pts, rng).params() for _ in range(restarts)])
    cost = cost_of(X)
 
    step = np.full(restarts, step0)
    fails = np.zeros(restarts, dtype=np.int64)
    history = np.empty(max_iter)
    cand = np.empty_like(X)
 
    # 1 iteration = 1 step for each restart, parallelized in numpy
    for start in range(0, max_iter, chunk):
        stop = min(start + chunk, max_iter)
        noise = rng.standard_normal((stop - start, restarts, 3))
 
        for i in range(start, stop):
            np.multiply(noise[i - start], step[:, None], out=cand)
            cand += X
            # radius stays non-negative
            np.abs(cand[:, 2], out=cand[:, 2])
            cand_cost = cost_of(cand)
 
            better = cand_cost < cost
            X[better] = cand[better]
            cost[better] = cand_cost[better]
 
            fails += 1
            fails[better] = 0
            stuck = fails >= patience
            # stuck restarts get a finer step
            step[stuck] *= 0.5
            fails[stuck] = 0
 
            history[i] = cost.min()

        # return the best circle, tightened to the farthest point
        k = int(cost.argmin())

    best = Circle(X[k, 0], X[k, 1], X[k, 2]).tighten(pts)
    return best, history



# function simulated_annealing(pts, max_iter, T0, alpha, step0, rng):
#     x  = random_start()
#     fx = cost(x)
#     best, fbest = x, fx
#     T = T0

#     for it in 1..max_iter:
#         step = step0 * (T / T0)             # a lépésköz a hűléssel együtt csökken
#         cand = x + rng.normal(0, step, size=2)
#         fc = cost(cand)
#         delta = fc - fx

#         if delta < 0 or rng.random() < exp(-delta / T):
#             x, fx = cand, fc

#         if fx < fbest:
#             best, fbest = x, fx

#         T = max(T * alpha, T_min)           # geometriai hűtés, alpha ≈ 0.995–0.999
#         history.append(fbest)

#     return best, fbest

# function genetic_algorithm(pts, pop_size, max_gen, pc, pm, sigma0, n_elite, k_tourn, rng):
#     pop = [random_start() for _ in 1..pop_size]          # shape (pop_size, 2)
#     fit = [cost(ind) for ind in pop]

#     for gen in 1..max_gen:
#         sigma = sigma0 * (1 - gen / max_gen)             # csökkenő mutáció (opcionális)
#         new_pop = top n_elite egyed (elitizmus)

#         while len(new_pop) < pop_size:
#             p1 = tournament(pop, fit, k_tourn)
#             p2 = tournament(pop, fit, k_tourn)

#             if rng.random() < pc:
#                 a = rng.random()
#                 child = a * p1 + (1 - a) * p2            # aritmetikai keresztezés
#             else:
#                 child = copy(p1)

#             if rng.random() < pm:
#                 child += rng.normal(0, sigma, size=2)    # Gauss-mutáció

#             child = clip(child, bounds)
#             new_pop.append(child)

#         pop = new_pop
#         fit = [cost(ind) for ind in pop]
#         history.append(min(fit))

#     return pop[argmin(fit)], min(fit)


# function tournament(pop, fit, k):
#     idx = k véletlen index
#     return pop[idx az, ahol fit minimális]

import numpy as np
from model.base import Circle


def random_start(pts, rng):
    """Random circle inside the bounding box of the data"""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = rng.uniform(lo, hi)
    r = rng.uniform(0.1, 0.5) * np.linalg.norm(hi - lo)
    return Circle(cx, cy, r)


def hill_climbing(pts, max_iter, step0, restarts, patience, rng):
    history = []
    best = None

    for _ in range(restarts):
        x = random_start(pts, rng)
        x.evaluate(pts)
        step, fails = step0, 0

        for _ in range(max_iter):
            p = x.params() + rng.normal(0, step, size=3)
            # radius must stay non-negative
            cand = Circle(p[0], p[1], abs(p[2]))
            cand.evaluate(pts)

            if cand.cost < x.cost:
                x = cand
                fails = 0
            else:
                fails += 1

            if fails >= patience:
                # stuck, so use a finer step
                step *= 0.5
                fails = 0

            if best is None or x.cost < best.cost:
                best = x
            history.append(best.cost)

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

import numpy as np
# function hill_climbing(pts, max_iter, step0, restarts, patience, rng):
def hill_climbing(pts,max_iter,step0,restarts,patience,rng):
    history = []
    global_best=None
    global_best_cost=None
#     global_best = None

#     for r in 1..restarts:
#         x  = random_start()
#         fx = cost(x)
#         step = step0
#         fails = 0

    for r in range(restarts):
        x=random_start()
        fx=cost(x)
        step=step0
        fails=0

        for iteration in range(max_iter):
            candidate=x+np.random.normal(0,step,size=2)
            fc=cost(candidate)

            if fc<fx:
                x,fx=candidate,fc
                fails=0
            else:
                fails+=1
            if fails>=patience:
                step*=0.5
                fails=0

            history.append(min(fx,global_best_cost))

        if global_best is None or fx<global_best_cost:
            global_best=x,fx
            global_best_cost=fx

    return global_best


#         for it in 1..max_iter:
#             cand = x + rng.normal(0, step, size=2)
#             fc = cost(cand)

#             if fc < fx:
#                 x, fx = cand, fc
#                 fails = 0
#             else:
#                 fails += 1

#             if fails >= patience:          # elakadt -> finomabb lépés
#                 step *= 0.5
#                 fails = 0

#             history.append(min(fx, global_best_cost))

#         if global_best is None or fx < global_best_cost:
#             global_best = x, fx

#     return global_best

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
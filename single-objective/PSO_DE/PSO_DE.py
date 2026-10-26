import numpy as np
import matplotlib.pyplot as plt
import opfunu

# PARTICLE SWARM OPTIMIZATION 
def pso(func, dim, lb, ub, pop_size=50, max_iter=1000,
        w=0.7, c1=1.5, c2=1.5, seed=None):
  
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # Khoi tao vi tri va van toc
    pos = rng.uniform(lb, ub, size=(pop_size, dim))
    vel_range = (ub - lb) * 0.1
    vel = rng.uniform(-vel_range, vel_range, size=(pop_size, dim))

    fitness = np.array([func(ind) for ind in pos])

    pbest_pos = pos.copy()
    pbest_val = fitness.copy()

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    convergence = np.empty(max_iter)

    for t in range(max_iter):
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        pos = pos + vel

        # Xu ly vi pham bien: clamp ve bien va trieu tieu van toc (phan xa nhe)
        below = pos < lb
        above = pos > ub
        pos = np.clip(pos, lb, ub)
        vel[below | above] *= -0.5

        fitness = np.array([func(ind) for ind in pos])

        improved = fitness < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fitness[improved]

        cur_best_idx = np.argmin(pbest_val)
        if pbest_val[cur_best_idx] < gbest_val:
            gbest_val = pbest_val[cur_best_idx]
            gbest_pos = pbest_pos[cur_best_idx].copy()

        convergence[t] = gbest_val

    return gbest_val, gbest_pos, convergence


# DIFFERENTIAL EVOLUTION (DE/rand/1/bin)
def de(func, dim, lb, ub, pop_size=50, max_iter=1000,
       F=0.5, CR=0.9, seed=None):

    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_pos = pop[best_idx].copy()
    best_val = fitness[best_idx]

    convergence = np.empty(max_iter)
    idx_all = np.arange(pop_size)

    for t in range(max_iter):
        for i in range(pop_size):
            # Chon 3 ca the khac nhau va khac i
            candidates = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            # --- Mutation (DE/rand/1) ---
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)  # xu ly vi pham bien

            # --- Crossover (binomial) ---
            cross_mask = rng.random(dim) <= CR
            j_rand = rng.integers(0, dim)
            cross_mask[j_rand] = True  # dam bao trial khac target it nhat 1 chieu
            trial = np.where(cross_mask, mutant, pop[i])

            # --- Selection (greedy) ---
            trial_val = func(trial)
            if trial_val <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_val
                if trial_val < best_val:
                    best_val = trial_val
                    best_pos = trial.copy()

        convergence[t] = best_val

    return best_val, best_pos, convergence

# CHAY THU NGHIEM TREN CEC2017 + VE BIEU DO HOI TU
def run_experiment(func_name="F12017", dim=10, pop_size=50, max_iter=300,
                    n_runs=5, seed=42):
    
    """
    Chay n_runs lan voi seed khac nhau, lay trung binh convergence curve.
    """
    
    func_class = opfunu.get_functions_by_classname(func_name)[0]
    func_obj = func_class(ndim=dim)
    func = func_obj.evaluate
    lb, ub = func_obj.lb, func_obj.ub

    pso_curves = np.empty((n_runs, max_iter))
    de_curves = np.empty((n_runs, max_iter))
    pso_finals, de_finals = [], []

    for r in range(n_runs):
        pso_val, _, pso_curve = pso(func, dim, lb, ub,
                                     pop_size=pop_size, max_iter=max_iter,
                                     seed=seed + r)
        de_val, _, de_curve = de(func, dim, lb, ub,
                                  pop_size=pop_size, max_iter=max_iter,
                                  seed=seed + r)
        pso_curves[r] = pso_curve
        de_curves[r] = de_curve
        pso_finals.append(pso_val)
        de_finals.append(de_val)

    pso_mean_curve = pso_curves.mean(axis=0)
    de_mean_curve = de_curves.mean(axis=0)

    print(f"Ham: {func_obj.name}  (f_global = {func_obj.f_global})")
    print(f"  PSO: mean = {np.mean(pso_finals):.4e}, std = {np.std(pso_finals):.4e}")
    print(f"  DE : mean = {np.mean(de_finals):.4e}, std = {np.std(de_finals):.4e}")

    # --- Ve bieu do hoi tu ---
    plt.figure(figsize=(8, 5))
    plt.plot(pso_mean_curve, label="PSO", linewidth=2)
    plt.plot(de_mean_curve, label="DE", linewidth=2)
    plt.yscale("log")
    plt.xlabel("The he (Iteration)")
    plt.ylabel("Gia tri fitness tot nhat (log scale)")
    plt.title(f"Bieu do hoi tu - {func_obj.name} (dim={dim}, trung binh {n_runs} lan chay)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = f"convergence_{func_name}_dim{dim}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Da luu bieu do: {out_path}")
    plt.close()

    return pso_finals, de_finals


if __name__ == "__main__":
    # F1  = Shifted and Rotated Bent Cigar (Unimodal)
    # F4  = Shifted and Rotated Rosenbrock (Simple Multimodal)
    # F6  = Shifted and Rotated Weierstrass (Simple Multimodal)
    for fname in ["F12017", "F42017", "F62017"]:
        run_experiment(func_name=fname, dim=10, pop_size=50,
                       max_iter=300, n_runs=5, seed=42)
        print("-" * 60)
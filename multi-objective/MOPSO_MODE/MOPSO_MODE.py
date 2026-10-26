import numpy as np
import matplotlib.pyplot as plt
from platypus import ZDT1, ZDT2, ZDT3, SMPSO, GDE3, nondominated

# Define mặt Pareto chuẩn 
def get_true_front(name, n=200):
    f1 = np.linspace(0, 1, n)
    if name == "ZDT1":
        return np.column_stack([f1, 1 - np.sqrt(f1)])
    elif name == "ZDT2":
        return np.column_stack([f1, 1 - f1**2])
    elif name == "ZDT3":
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        # Giữ lại các đoạn không bị trội của ZDT3
        pts = np.column_stack([f1, f2])
        mask = np.ones(n, dtype=bool)
        for i, p in enumerate(pts):
            if any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)):
                mask[i] = False
        return pts[mask]

# Tính IGD 
def calc_igd(obtained, true_front):
    # dists: ma trận khoảng cách giữa các điểm true_front và obtained
    dists = np.linalg.norm(true_front[:, None, :] - obtained[None, :, :], axis=2)
    return np.mean(np.min(dists, axis=1))

problems = {"ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3}
results = {}

for pname, PClass in problems.items():
    print(f"\n--- {pname} ---")
    ref = get_true_front(pname)

    # Tạo instance problem mới cho mỗi thuật toán
    algos = {
        "MOPSO (SMPSO)": SMPSO(PClass(), swarm_size=100, leader_size=100),
        "MODE (GDE3)":   GDE3(PClass(), population_size=100, de_rate=0.5, de_stepsize=0.5)
    }

    for aname, algo in algos.items():
        algo.run(15000)  # Số lượt đánh giá
        sols = nondominated(algo.result)
        objs = np.array([s.objectives for s in sols])
        results[(pname, aname)] = objs

        igd_val = calc_igd(objs, ref)
        print(f"{aname:15s} | Số nghiệm: {len(objs):3d} | IGD = {igd_val:.5f}")

# Trực quan hóa kết quả
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, pname in zip(axes, problems.keys()):
    ref = get_true_front(pname)
    ax.scatter(ref[:, 0], ref[:, 1], c='gray', s=8, alpha=0.5, label="Pareto True")
    
    for aname in ["MOPSO (SMPSO)", "MODE (GDE3)"]:
        pts = results[(pname, aname)]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, label=aname)

    ax.set_title(pname)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
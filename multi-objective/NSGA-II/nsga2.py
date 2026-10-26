import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# TẠO DỮ LIỆU BÀI TOÁN 
np.random.seed(1)

N_CITIES = 20
coords = np.random.rand(N_CITIES, 2) * 100
SPEED = 1.0

dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

avg_edge_time = dist_matrix[dist_matrix > 0].mean() / SPEED
expected_tour_time = N_CITIES * avg_edge_time
due_time = np.random.rand(N_CITIES) * expected_tour_time * 0.8


def route_objectives(route):
    """Tính (f1: tổng quãng đường, f2: tổng thời gian chờ) cho 1 hành trình (permutation)."""
    total_distance = 0.0
    total_wait = 0.0
    current_time = 0.0

    for i in range(len(route)):
        a = route[i]
        b = route[(i + 1) % len(route)]
        d = dist_matrix[a, b]
        total_distance += d
        current_time += d / SPEED

        wait = max(0.0, due_time[b] - current_time)
        total_wait += wait
        current_time += wait

    return total_distance, total_wait


class TSPMultiObjective(ElementwiseProblem):
    def __init__(self, n_cities):
        super().__init__(n_var=n_cities, n_obj=2, n_constr=0, xl=0, xu=n_cities - 1, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = route_objectives(x)
        out["F"] = [f1, f2]


problem = TSPMultiObjective(N_CITIES)

sampling = PermutationRandomSampling()
crossover = OrderCrossover()
mutation = InversionMutation()

POP_SIZE = 100
N_GEN = 150
N_SEEDS = 10          # số lần chạy độc lập cho mỗi thuật toán
SEEDS = range(1, N_SEEDS + 1)

ref_dirs = get_reference_directions("uniform", 2, n_partitions=POP_SIZE - 1)


def run_nsga2(seed):
    algo = NSGA2(pop_size=POP_SIZE, sampling=sampling, crossover=crossover,
                 mutation=mutation, eliminate_duplicates=True)
    res = minimize(problem, algo, termination=("n_gen", N_GEN), seed=seed, verbose=False)
    return res.F


def run_moead(seed):
    algo = MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7,
                 sampling=sampling, crossover=crossover, mutation=mutation)
    res = minimize(problem, algo, termination=("n_gen", N_GEN), seed=seed, verbose=False)
    return res.F


# CHẠY N_SEEDS LẦN CHO MỖI THUẬT TOÁN, LƯU LẠI TOÀN BỘ KẾT QUẢ
F_nsga2_runs = []   # danh sách F (mảng nghiệm) của từng lần chạy NSGA-II
F_moead_runs = []

for seed in SEEDS:
    F_nsga2_runs.append(run_nsga2(seed))
    F_moead_runs.append(run_moead(seed))
    print(f"Đã chạy xong seed={seed}")

# REFERENCE POINT DÙNG CHUNG CHO TẤT CẢ CÁC LẦN CHẠY
all_F = np.vstack(F_nsga2_runs + F_moead_runs)
ref_point = all_F.max(axis=0) * 1.1

# Pareto front chung ("tốt nhất từng thấy") để biết mỗi thuật toán đóng góp bao nhiêu
nd_idx = NonDominatedSorting().do(all_F, only_non_dominated_front=True)
pareto_F = all_F[nd_idx]

n_total_nsga2 = sum(len(F) for F in F_nsga2_runs)
n_total_moead = sum(len(F) for F in F_moead_runs)
# đếm xem trong pareto_F, bao nhiêu điểm gốc là của NSGA-II hay của MOEA/D
boundary = sum(len(F) for F in F_nsga2_runs)  # số hàng đầu trong all_F thuộc về NSGA-II
n_contrib_nsga2 = np.sum(nd_idx < boundary)
n_contrib_moead = np.sum(nd_idx >= boundary)

# TÍNH HV CHO TỪNG LẦN CHẠY, DÙNG CHUNG 1 REFERENCE POINT
hv_indicator = HV(ref_point=ref_point)
hv_nsga2_list = [hv_indicator(F) for F in F_nsga2_runs]
hv_moead_list = [hv_indicator(F) for F in F_moead_runs]

mean_nsga2, std_nsga2 = np.mean(hv_nsga2_list), np.std(hv_nsga2_list)
mean_moead, std_moead = np.mean(hv_moead_list), np.std(hv_moead_list)

print("=" * 60)
print(f"Số lần chạy mỗi thuật toán : {N_SEEDS}")
print(f"Kích thước Pareto front chung của cả 2 thuật toán : {len(pareto_F)}")
print(f"  - Đóng góp từ NSGA-II : {n_contrib_nsga2} điểm")
print(f"  - Đóng góp từ MOEA/D  : {n_contrib_moead} điểm")
print("-" * 60)
print(f"NSGA-II : HV trung bình = {mean_nsga2:.2f}  (độ lệch chuẩn = {std_nsga2:.2f})")
print(f"MOEA/D  : HV trung bình = {mean_moead:.2f}  (độ lệch chuẩn = {std_moead:.2f})")
print("=" * 60)
if mean_nsga2 > mean_moead:
    print("=> NSGA-II có HV trung bình cao hơn -> bao phủ/hội tụ tốt hơn trên bài toán này.")
elif mean_moead > mean_nsga2:
    print("=> MOEA/D có HV trung bình cao hơn -> bao phủ/hội tụ tốt hơn trên bài toán này.")
else:
    print("=> Hai thuật toán cho HV trung bình bằng nhau.")

# 5. BIỂU ĐỒ: (a) so sánh phân bố HV qua các seed, (b) biên Pareto của lần chạy tốt nhất
best_nsga2_idx = int(np.argmax(hv_nsga2_list))
best_moead_idx = int(np.argmax(hv_moead_list))

plt.figure(figsize=(9, 7))

# Vẽ Pareto front chung NẰM DƯỚI CÙNG
plt.scatter(pareto_F[:, 0], pareto_F[:, 1], 
            s=60, color="lightgray", edgecolors="gray", alpha=0.8,
            label="Pareto front chung", zorder=1)

# Vẽ NSGA-II 
plt.scatter(F_nsga2_runs[best_nsga2_idx][:, 0], F_nsga2_runs[best_nsga2_idx][:, 1],
            s=35, alpha=0.7, color="#1f77b4",
            label=f"NSGA-II (HV tốt nhất={hv_nsga2_list[best_nsga2_idx]:.0f})", zorder=2)

# Vẽ MOEA/D
plt.scatter(F_moead_runs[best_moead_idx][:, 0], F_moead_runs[best_moead_idx][:, 1],
            s=35, alpha=0.7, marker="^", color="#ff7f0e",
            label=f"MOEA/D (HV tốt nhất={hv_moead_list[best_moead_idx]:.0f})", zorder=3)

plt.xlabel("f1: Tổng quãng đường")
plt.ylabel("f2: Tổng thời gian chờ")
plt.title("Biên Pareto (lần chạy tốt nhất mỗi thuật toán)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pareto_front_comparison.png", dpi=150)
plt.show()
print("Đã lưu biểu đồ: pareto_front_comparison.png")
import math
import random
from collections import defaultdict

# ==========================================
# 1. TIỆN ÍCH TÍNH TOÁN & ĐỒ THỊ
# ==========================================

def compute_dist_matrix(coords):
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = round(math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]))
            dist[i][j] = d
            dist[j][i] = d
    return dist

def tour_cost(tour, dist):
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

def tour_to_edges(tour):
    n = len(tour)
    return {tuple(sorted((tour[i], tour[(i + 1) % n]))) for i in range(n)}

# Khởi tạo quần thể kèm Local Search 2-Opt nhanh
def local_search_2opt(tour, dist):
    n = len(tour)
    improved = True
    best_tour = list(tour)
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u1, v1 = best_tour[i], best_tour[i + 1]
                u2, v2 = best_tour[j], best_tour[(j + 1) % n]
                if dist[u1][u2] + dist[v1][v2] < dist[u1][v1] + dist[u2][v2]:
                    best_tour[i + 1:j + 1] = reversed(best_tour[i + 1:j + 1])
                    improved = True
                    break
            if improved:
                break
    return best_tour

# ==========================================
# 2. TOÁN TỬ LAI GHÉP NAGATA EAX (CLEAN)
# ==========================================

def get_ab_cycles(tour_a, tour_b):
    edges_a = tour_to_edges(tour_a)
    edges_b = tour_to_edges(tour_b)
    diff_a = list(edges_a - edges_b)
    diff_b = list(edges_b - edges_a)

    if not diff_a or not diff_b:
        return []

    adj_a = defaultdict(list)
    adj_b = defaultdict(list)
    for u, v in diff_a:
        adj_a[u].append(v); adj_a[v].append(u)
    for u, v in diff_b:
        adj_b[u].append(v); adj_b[v].append(u)

    unused_a = set(diff_a)
    unused_b = set(diff_b)
    ab_cycles = []

    while unused_a and unused_b:
        start_edge = next(iter(unused_a))
        start = start_edge[0]
        curr = start
        path = [curr]
        modes = []
        mode = 'A'

        while True:
            adj = adj_a if mode == 'A' else adj_b
            unused = unused_a if mode == 'A' else unused_b
            candidates = [nxt for nxt in adj[curr] if tuple(sorted((curr, nxt))) in unused]

            if not candidates:
                # Xóa sạch các cạnh đã duyệt dở để tránh vòng lặp vô tận
                for idx in range(len(path) - 1):
                    e = tuple(sorted((path[idx], path[idx + 1])))
                    unused_a.discard(e)
                    unused_b.discard(e)
                break

            nxt = random.choice(candidates)
            edge = tuple(sorted((curr, nxt)))
            unused.discard(edge)

            if nxt in path:
                idx = path.index(nxt)
                c_nodes = path[idx:] + [nxt]
                c_modes = modes[idx:] + [mode]
                cyc_a = []
                cyc_b = []
                for k in range(len(c_nodes) - 1):
                    e = tuple(sorted((c_nodes[k], c_nodes[k + 1])))
                    if c_modes[k] == 'A':
                        cyc_a.append(e)
                    else:
                        cyc_b.append(e)

                # Điều kiện hợp lệ: số cạnh A phải bằng số cạnh B
                if len(cyc_a) >= 2 and len(cyc_a) == len(cyc_b):
                    ab_cycles.append({'A': cyc_a, 'B': cyc_b})
                break

            path.append(nxt)
            modes.append(mode)
            curr = nxt
            mode = 'B' if mode == 'A' else 'A'

    return ab_cycles

def reconnect_subtours(subtours, dist, n):
    while len(subtours) > 1:
        # Chọn subtour ngắn nhất để ghép nối trước (chuẩn quy tắc Step 5)
        subtours.sort(key=len)
        main_sub = subtours.pop(0)

        best_gain = float('inf')
        target_sub_idx = -1
        split_m = (-1, -1)
        split_s = (-1, -1)

        len_m = len(main_sub)
        for s_idx, target_sub in enumerate(subtours):
            len_s = len(target_sub)
            for i in range(len_m):
                u1, u2 = main_sub[i], main_sub[(i + 1) % len_m]
                for j in range(len_s):
                    v1, v2 = target_sub[j], target_sub[(j + 1) % len_s]

                    # Nối thẳng (u1 - v1) & (u2 - v2)
                    gain1 = dist[u1][v1] + dist[u2][v2] - dist[u1][u2] - dist[v1][v2]
                    if gain1 < best_gain:
                        best_gain = gain1
                        target_sub_idx = s_idx
                        split_m = (i, (i + 1) % len_m)
                        split_s = (j, (j + 1) % len_s)

        chosen_sub = subtours.pop(target_sub_idx)
        i1, i2 = split_m
        j1, j2 = split_s

        merged = (
            main_sub[:i1 + 1] +
            chosen_sub[j1::-1] + chosen_sub[:j1:-1] +
            main_sub[i2:]
        )
        subtours.insert(0, merged)

    return subtours[0]

def eax_crossover(parent_a, parent_b, dist):
    n = len(parent_a)
    cycles = get_ab_cycles(parent_a, parent_b)
    if not cycles:
        return list(parent_a)

    # Chọn 1 chu trình tốt nhất (giảm trọng số cao nhất)
    def cycle_gain(cyc):
        return sum(dist[u][v] for u, v in cyc['B']) - sum(dist[u][v] for u, v in cyc['A'])

    best_cycle = min(cycles, key=cycle_gain)

    # Tạo tập cạnh trung gian E_C
    ec_edges = tour_to_edges(parent_a)
    for e in best_cycle['A']:
        ec_edges.discard(e)
    for e in best_cycle['B']:
        ec_edges.add(e)

    # Xây dựng danh sách kề để bóc tách subtours
    adj = defaultdict(list)
    for u, v in ec_edges:
        adj[u].append(v); adj[v].append(u)

    visited = set()
    subtours = []
    for node in range(n):
        if node not in visited and adj[node]:
            st = []
            curr = node
            prev = -1
            while curr not in visited:
                visited.add(curr)
                st.append(curr)
                nbrs = adj[curr]
                nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else nbrs[0])
                prev, curr = curr, nxt
            subtours.append(st)

    if not subtours:
        return list(parent_a)

    return reconnect_subtours(subtours, dist, n)

# ==========================================
# 3. GA MAIN ENGINE
# ==========================================

def run_ga_eax(coords, pop_size=20, generations=50):
    n = len(coords)
    dist = compute_dist_matrix(coords)

    # Khởi tạo quần thể
    population = []
    for _ in range(pop_size):
        tour = list(range(n))
        random.shuffle(tour)
        population.append(local_search_2opt(tour, dist))

    best_tour = min(population, key=lambda t: tour_cost(t, dist))
    best_cost = tour_cost(best_tour, dist)
    print(f"Chi phí khởi đầu (sau 2-Opt): {best_cost}")

    # Vòng lặp tiến hóa
    for gen in range(generations):
        offspring_pool = []
        for _ in range(pop_size):
            p1, p2 = random.sample(population, 2)
            child = eax_crossover(p1, p2, dist)
            offspring_pool.append(child)

        # Chọn lọc tự nhiên (Elitism)
        population = sorted(population + offspring_pool, key=lambda t: tour_cost(t, dist))[:pop_size]

        current_best = population[0]
        current_cost = tour_cost(current_best, dist)

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_best
            print(f"Thế hệ {gen + 1:02d}: Đạt mốc Cost mới = {best_cost}")

    return best_tour, best_cost

# ==========================================
# 4. CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    sample_cities = [
        (60, 200), (180, 200), (80, 180), (140, 180), (20, 160),
        (100, 160), (200, 160), (140, 140), (40, 120), (100, 120),
        (180, 100), (60, 80), (120, 80), (180, 60), (20, 40),
        (100, 40), (200, 40), (60, 20), (140, 20), (160, 120)
    ]

    best_route, min_distance = run_ga_eax(sample_cities, pop_size=20, generations=30)
    print(f"\nCost tối ưu tìm được: {min_distance}")
    print(f"Lộ trình: {best_route}")
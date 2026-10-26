import math
import random
import time

EPS = 1e-10

class TSP:
    def __init__(self, points):
        self.points = list(points)
        self.n = len(points)
        if self.n < 3:
            raise ValueError('At least three points required')
        self.D = [[math.dist(a, b) for b in self.points] for a in self.points]
        self.order = [sorted((j for j in range(self.n) if j != i),
                             key=lambda j: (self.D[i][j], j)) for i in range(self.n)]

    def length(self, tour):
        return sum(self.D[tour[i]][tour[(i + 1) % len(tour)]]
                   for i in range(len(tour)))

    @staticmethod
    def validate(tour, expected):
        if len(tour) != len(expected) or len(set(tour)) != len(tour) or set(tour) != set(expected):
            raise AssertionError(f'Invalid tour: {tour}; expected {expected}')

    def near(self, city, included, p):
        if p < 1:
            raise ValueError('p must be >= 1')
        return self._near_cached(city, frozenset(included), p)

    def _near_cached(self, city, included, p):
        return [u for u in self.order[city] if u in included][:p]

    def geni_insert(self, v, tour, p):
        """Best admissible Type I/II GENI insertion; cheapest insertion as fallback."""
        m = len(tour)
        if v in tour:
            raise ValueError('City already in tour')
        included = set(tour)
        target = included | {v}
        best, best_cost = None, math.inf

        def consider(t):
            nonlocal best, best_cost
            if len(t) != m + 1 or set(t) != target or len(set(t)) != m + 1:
                return
            cost = self.length(t)
            if cost < best_cost - EPS:
                best, best_cost = t, cost

        # Standard insertion is permitted only between consecutive vertices
        # that both belong to N_p(v), as in GENI (1992).
        npv = self.near(v, included, p)
        for i in range(m):
            if tour[i] in npv and tour[(i+1) % m] in npv:
                consider(tour[:i+1] + [v] + tour[i+1:])
        # For the initial three-city tour, all three vertices are eligible.
        if m < 5:
            if best is None:
                raise RuntimeError("No admissible insertion")
            return best

        for orient in (tour, tour[::-1]):
            for vi in npv:
                i = orient.index(vi)
                R = orient[i:] + orient[:i]
                pos = {u: j for j, u in enumerate(R)}
                nk = self.near(R[1], included, p)
                for vj in npv:
                    pj = pos[vj]
                    if not 1 <= pj <= m-2:
                        continue
                    # GENI Type I (Figure 1, 1992).
                    for vk in nk:
                        pk = pos[vk]
                        if pj+1 <= pk <= m-1:
                            consider([R[0], v] + R[1:pj+1][::-1]
                                     + R[pj+1:pk+1][::-1] + R[pk+1:])
                    # GENI Type II (Figure 2, 1992).
                    nl = self.near(R[pj+1], included, p)
                    for vl in nl:
                        pl = pos[vl]
                        if not 2 <= pl <= pj:
                            continue
                        for vk in nk:
                            pk = pos[vk]
                            if pj+2 <= pk <= m-1:
                                consider([R[0], v] + R[pl:pj+1][::-1]
                                         + R[pj+1:pk] + R[1:pl][::-1] + R[pk:])
        if best is None:
            raise RuntimeError("No admissible GENI insertion found")
        self.validate(best, target)
        return best

    def geni(self, p=6, rng=None):
        rng = rng if rng is not None else random.Random()
        cities = list(range(self.n))
        rng.shuffle(cities)
        tour = cities[:3]
        for v in cities[3:]:
            tour = self.geni_insert(v, tour, p)
            self.validate(tour, set(cities[:len(tour)]))
        return tour

    def unstring_candidates(self, v, tour, p):
        """Enumerate Type I/II reverse reconnections 
        """
        m = len(tour)
        if v not in tour:
            raise ValueError('City not in tour')
        remaining = set(tour) - {v}
        seen = set()

        def emit(candidate):
            if len(candidate) != m-1 or len(set(candidate)) != m-1 or set(candidate) != remaining:
                return None
            # Undirected tours are equivalent under rotation/reversal.
            anchor = min(candidate)
            q = candidate.index(anchor)
            canonical = candidate[q:] + candidate[:q]
            reverse = [canonical[0]] + canonical[:0:-1]
            key = min(tuple(canonical), tuple(reverse))
            if key in seen:
                return None
            seen.add(key)
            return candidate

        for orient in (tour, tour[::-1]):
            ix = (orient.index(v) - 1) % m
            R = orient[ix:] + orient[:ix]  # R[0] predecessor, R[1] == v
            nb = {}
            def N(c):
                if c not in nb:
                    nb[c] = set(self.near(c, remaining, p))
                return nb[c]
            # Ordinary unstringing is a degenerate reconnection.
            t = emit([R[0]] + R[2:])
            if t is not None:
                yield t
            if m < 6:
                continue
            # Reverse Type I: two path reversals.
            for a in range(1, m-2):
                if R[2+a] not in N(R[1+a]):
                    continue
                for b in range(a+1, m-1):
                    t = emit([R[0]] + R[2:2+a][::-1]
                             + R[2+a:2+b][::-1] + R[2+b:])
                    if t is not None:
                        yield t
            # Reverse Type II: three path blocks reconnected.
            valid_s2 = [s2 for s2 in range(1, m-3)
                        if R[1+s2] in N(R[2+s2])]
            valid_e = [e for e in range(5, m+1)
                       if R[e % m] in N(R[e-1])]
            for s2 in valid_s2:
                for e in valid_e:
                    for s3 in range(1, e-2-s2):
                        q = 2+s2+s3
                        if q >= e:
                            continue
                        t = emit([R[0]] + R[q:e][::-1]
                                 + R[2:2+s2][::-1] + R[2+s2:q] + R[e:])
                        if t is not None:
                            yield t

    def us(self, tour, p, max_passes=None, max_seconds=None):
        """GENIUS US: scan vertices, unstring (I/II), reinsert (I/II), repeat.
        """
        current = list(tour)
        self.validate(current, set(range(self.n)))
        passes = 0
        deadline = time.monotonic() + max_seconds if max_seconds is not None else None
        while max_passes is None or passes < max_passes:
            if deadline is not None and time.monotonic() >= deadline:
                break
            passes += 1
            improved = False
            for v in list(current):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                current_cost = self.length(current)
                best = current
                best_cost = current_cost
                for reduced in self.unstring_candidates(v, current, p):
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    try:
                        candidate = self.geni_insert(v, reduced, p)
                    except RuntimeError:
                        continue  # Khong co phep chen hop le voi k hien tai
                    score = self.length(candidate)
                    if score < best_cost - EPS:
                        best, best_cost = candidate, score
                if best_cost < current_cost - EPS:
                    current = best
                    improved = True
            if not improved:
                break
        self.validate(current, set(range(self.n)))
        return current


def genius(tsp, p=6, rng=None, start=None):
    initial = list(start) if start is not None else tsp.geni(p, rng)
    return tsp.us(initial, p)


def g_vns(tsp, p=6, rng=None, start=None, max_seconds=None):
    """US(k), k=2..floor(p/2)+1."""
    if p < 2:
        raise ValueError('G+VNS requires p >= 2')
    best = list(start) if start is not None else tsp.geni(p, rng)
    tsp.validate(best, set(range(tsp.n)))
    f_best = tsp.length(best)
    k, k_max = 2, p//2+1
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    while k <= k_max:
        if deadline is not None and time.monotonic() >= deadline:
            break
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        candidate = tsp.us(best, k, max_seconds=remaining)
        value = tsp.length(candidate)
        if value < f_best - EPS:
            best, f_best, k = candidate, value, 2
        else:
            k += 1
    return best, f_best


def self_test():
    for n in (6, 8, 12):
        for seed in (0,):
            rng = random.Random(1000*n+seed)
            tsp = TSP([(rng.random()*100, rng.random()*100) for _ in range(n)])
            start = tsp.geni(6, rng)
            tsp.validate(start, set(range(n)))
            for v in start:
                for reduced in tsp.unstring_candidates(v, start, 6):
                    tsp.validate(reduced, set(range(n))-{v})
                    tsp.validate(tsp.geni_insert(v, reduced, 6), set(range(n)))
            a = genius(tsp, 6, start=start)
            b, f = g_vns(tsp, 6, start=start)
            assert tsp.length(a) <= tsp.length(start)+EPS
            assert f <= tsp.length(start)+EPS
            assert abs(f-tsp.length(b)) < EPS
    print('Self-tests passed')


if __name__ == '__main__':
    # Chay nhanh truoc; tang N sau khi do thoi gian.
    N, P, RUNS = 50, 6, 3
    LIMIT_SECONDS = 15  # Gioi han thuc dung, KHONG phai dieu kien dung cua paper
    for seed in range(RUNS):
        rng = random.Random(seed)
        tsp = TSP([(rng.random()*100, rng.random()*100) for _ in range(N)])
        print(f"Run {seed+1}: GENI...", flush=True)
        start = tsp.geni(P, rng)
        print("GENIUS...", flush=True)
        baseline = tsp.us(start, P, max_seconds=LIMIT_SECONDS)
        print("G+VNS...", flush=True)
        best, score = g_vns(tsp, P, start=start, max_seconds=LIMIT_SECONDS)
        print(f"GENI={tsp.length(start):.2f} GENIUS={tsp.length(baseline):.2f} G+VNS={score:.2f}")

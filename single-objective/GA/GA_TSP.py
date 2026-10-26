import random
import math

def calculate_distance_matrix(cities):
    n = len(cities)
    distance = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                distance[i][j] = math.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2
                )
    return distance

def calculate_cost(route, distance):
    cost = 0
    for i in range(len(route) - 1):
        cost += distance[route[i]][route[i + 1]]
    cost += distance[route[-1]][route[0]]
    return cost

def create_individual(n):
    route = list(range(n))
    random.shuffle(route)
    return route

def create_population(population_size, n):
    population = []
    for _ in range(population_size):
        individual = create_individual(n)
        population.append(individual)
    return population

def fitness(individual, distance):
    cost = calculate_cost(individual, distance)
    return 1 / cost

def tournament_selection(population, distance, tournament_size=3):
    candidates = random.sample(
        population,
        tournament_size
    )
    best = min(
        candidates,
        key=lambda x: calculate_cost(x, distance)
    )
    return best.copy()

def crossover(parent1, parent2):
    n = len(parent1)
    start, end = sorted(
        random.sample(range(n), 2)
    )
    child = [None] * n
    child[start:end] = parent1[start:end]
    remaining = [
        city for city in parent2
        if city not in child
    ]
    index = 0
    for i in range(n):
        if child[i] is None:
            child[i] = remaining[index]
            index += 1
    return child

def mutation(route, mutation_rate=0.02):
    route = route.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(
            range(len(route)),
            2
        )
        route[i], route[j] = route[j], route[i]
    return route

def evolutionary_algorithm(
    cities,
    population_size=100,
    generations=500,
    mutation_rate=0.02,
    elite_size=2
):
    n = len(cities)
    distance = calculate_distance_matrix(cities)
    population = create_population(
        population_size,
        n
    )
    best_route = None
    best_cost = float("inf")
    for generation in range(generations):
        population.sort(
            key=lambda x: calculate_cost(x, distance)
        )
        current_best = population[0]
        current_cost = calculate_cost(
            current_best,
            distance
        )
        if current_cost < best_cost:

            best_cost = current_cost
            best_route = current_best.copy()
        new_population = []
        for i in range(elite_size):
            new_population.append(
                population[i].copy()
            )
        while len(new_population) < population_size:
            parent1 = tournament_selection(
                population,
                distance
            )
            parent2 = tournament_selection(
                population,
                distance
            )
            child = crossover(
                parent1,
                parent2
            )
            child = mutation(
                child,
                mutation_rate
            )
            new_population.append(child)
        population = new_population
        if generation % 50 == 0:
            print(
                f"Generation {generation}: "
                f"Best cost = {best_cost:.2f}"
            )
    return best_route, best_cost

if __name__ == "__main__":
    cities = [
        (0, 0),
        (2, 6),
        (5, 3),
        (8, 7),
        (6, 1),
        (1, 4),
        (7, 5),
        (3, 8)
    ]

    best_route, best_cost = evolutionary_algorithm(
        cities,
        population_size=100,
        generations=500,
        mutation_rate=0.02,
        elite_size=2
    )
    print("Best route:")

    # In thành phố bắt đầu lại ở cuối
    route_display = best_route + [best_route[0]]

    print(" -> ".join(
        str(city) for city in route_display
    ))

    print(f"Best cost: {best_cost:.2f}")
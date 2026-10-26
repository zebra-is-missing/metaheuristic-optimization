import random

def calculate_fitness(chromosome, weights, values, capacity):
    total_weight = 0
    total_value = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            total_weight += weights[i]
            total_value += values[i]
    if total_weight > capacity:
        return 0, total_weight, total_value
    return total_value, total_weight, total_value

def create_chromosome(n):
    return [
        random.randint(0, 1)
        for _ in range(n)
    ]

def create_population(population_size, n):
    return [
        create_chromosome(n)
        for _ in range(population_size)
    ]

def selection(
    population,
    weights,
    values,
    capacity,
    tournament_size=3
):
    tournament = random.sample(
        population,
        tournament_size
    )
    return max(
        tournament,
        key=lambda chromosome:
            calculate_fitness(
                chromosome,
                weights,
                values,
                capacity
            )[0]
    )

def crossover(parent1, parent2):
    point = random.randint(
        1,
        len(parent1) - 1
    )
    child = (
        parent1[:point]
        + parent2[point:]
    )
    return child

def mutation(chromosome, mutation_rate=0.02):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# Genetic Algorithm
def genetic_algorithm_knapsack(
    weights,
    values,
    capacity,
    population_size=100,
    generations=500,
    mutation_rate=0.02
):
    n = len(weights)

    population = create_population(population_size, n)

    best_chromosome = None
    best_value = 0
    best_weight = 0

    for generation in range(generations):
        current_best = max(
            population,
            key=lambda chromosome: calculate_fitness(
                chromosome, weights, values, capacity
            )[0]
        )

        current_value, current_weight, _ = calculate_fitness(
            current_best, weights, values, capacity
        )

        if current_value > best_value:
            best_value = current_value
            best_weight = current_weight
            best_chromosome = current_best[:]

        new_population = []

        new_population.append(
            best_chromosome[:]
        )

        while len(new_population) < population_size:
            parent1 = selection(
                population,
                weights,
                values,
                capacity
            )
            parent2 = selection(
                population,
                weights,
                values,
                capacity
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
    return (
        best_chromosome,
        best_weight,
        best_value
    )

weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 5

best_chromosome, best_weight, best_value = \
    genetic_algorithm_knapsack(
        weights,
        values,
        capacity,
        population_size=100,
        generations=500,
        mutation_rate=0.02
    )

print("Best chromosome:", best_chromosome)
print("Total weight:", best_weight)
print("Total value:", best_value)
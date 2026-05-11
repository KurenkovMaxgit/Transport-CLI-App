from __future__ import annotations

import random
import time

from ..evaluator import solve_for_modes
from ..models import ProblemInstance, SolveResult


def _random_chromosome(problem: ProblemInstance, rng: random.Random) -> list[int]:
    supply_part = [rng.randrange(problem.k) for _ in range(problem.m)]
    demand_part = [rng.randrange(problem.l) for _ in range(problem.n)]
    return supply_part + demand_part


def _split_chromosome(
    problem: ProblemInstance, chromosome: list[int]
) -> tuple[list[int], list[int]]:
    return chromosome[: problem.m], chromosome[problem.m :]


def _evaluate(
    problem: ProblemInstance,
    chromosome: list[int],
    cache: dict[tuple[int, ...], SolveResult],
) -> SolveResult:
    key = tuple(chromosome)
    if key not in cache:
        supply_modes, demand_modes = _split_chromosome(problem, chromosome)
        cache[key] = solve_for_modes(
            problem,
            supply_modes,
            demand_modes,
            algorithm_name="Генетичний алгоритм",
        )
    return cache[key]


def _tournament_select(
    population: list[list[int]],
    fitness: list[float],
    tournament_size: int,
    rng: random.Random,
) -> list[int]:
    indexes = rng.sample(
        range(len(population)), k=min(tournament_size, len(population))
    )
    best_idx = min(indexes, key=lambda idx: fitness[idx])
    return population[best_idx][:]


def _crossover(
    parent1: list[int], parent2: list[int], rng: random.Random
) -> tuple[list[int], list[int]]:
    if len(parent1) <= 1:
        return parent1[:], parent2[:]
    point = rng.randrange(1, len(parent1))
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def _mutate(
    problem: ProblemInstance,
    chromosome: list[int],
    mutation_prob: float,
    rng: random.Random,
) -> list[int]:
    child = chromosome[:]
    if rng.random() >= mutation_prob:
        return child

    gene_idx = rng.randrange(len(child))

    if gene_idx < problem.m:
        possible = [x for x in range(problem.k) if x != child[gene_idx]]
    else:
        possible = [x for x in range(problem.l) if x != child[gene_idx]]

    if possible:
        child[gene_idx] = rng.choice(possible)

    return child


def genetic_solve(
    problem: ProblemInstance,
    population_size: int = 40,
    max_stall_gen: int = 30,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.15,
    tournament_size: int = 3,
    seed: int | None = None,
) -> SolveResult:
    start_total = time.perf_counter()
    rng = random.Random(seed)
    cache: dict[tuple[int, ...], SolveResult] = {}

    population = [_random_chromosome(problem, rng) for _ in range(population_size)]

    best_chromosome: list[int] | None = None
    best_result: SolveResult | None = None
    best_cost = float("inf")
    stall = 0
    generation = 0
    history_best: list[float] = []
    history_iteration_best: list[float] = []

    while stall < max_stall_gen:
        results = [_evaluate(problem, chromosome, cache) for chromosome in population]
        fitness = [result.total_cost for result in results]

        iteration_best_idx = min(range(len(population)), key=lambda idx: fitness[idx])
        iteration_best_cost = fitness[iteration_best_idx]
        history_iteration_best.append(iteration_best_cost)

        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_chromosome = population[iteration_best_idx][:]
            best_result = results[iteration_best_idx]
            stall = 0
        else:
            stall += 1

        history_best.append(best_cost)

        new_population: list[list[int]] = []

        if best_chromosome is not None:
            new_population.append(best_chromosome[:])

        while len(new_population) < population_size:
            p1 = _tournament_select(population, fitness, tournament_size, rng)
            p2 = _tournament_select(population, fitness, tournament_size, rng)

            if rng.random() < crossover_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = _mutate(problem, c1, mutation_prob, rng)
            c2 = _mutate(problem, c2, mutation_prob, rng)

            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population
        generation += 1

    if best_result is None or best_chromosome is None:
        raise RuntimeError("Генетичний алгоритм не знайшов жодного розв'язку.")

    end_total = time.perf_counter()
    supply_modes, demand_modes = _split_chromosome(problem, best_chromosome)

    return SolveResult(
        algorithm_name="Генетичний алгоритм",
        supply_modes=supply_modes,
        demand_modes=demand_modes,
        supply=best_result.supply,
        demand=best_result.demand,
        supply_ext=best_result.supply_ext,
        demand_ext=best_result.demand_ext,
        costs_ext=best_result.costs_ext,
        plan_ext=best_result.plan_ext,
        dummy_type=best_result.dummy_type,
        total_cost=best_result.total_cost,
        runtime_sec=end_total - start_total,
        extra={
            "best_chromosome": best_chromosome,
            "generations": generation,
            "cache_size": len(cache),
            "history_best": history_best,
            "history_iteration_best": history_iteration_best,
            "population_size": population_size,
            "max_stall_gen": max_stall_gen,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "tournament_size": tournament_size,
        },
    )

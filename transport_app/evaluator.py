from __future__ import annotations

import time

from .balancing import balance_transport_problem, select_supply_demand
from .models import ProblemInstance, SolveResult
from .vogel import vogel_method


def calculate_cost(plan: list[list[float]], costs: list[list[float]]) -> float:
    total = 0.0
    for i, row in enumerate(plan):
        for j, amount in enumerate(row):
            total += amount * costs[i][j]
    return total


def solve_for_modes(
    problem: ProblemInstance,
    supply_modes: list[int],
    demand_modes: list[int],
    algorithm_name: str = "Розв'язання за режимами",
    extra: dict | None = None,
) -> SolveResult:
    start = time.perf_counter()

    supply, demand = select_supply_demand(problem, supply_modes, demand_modes)
    supply_ext, demand_ext, costs_ext, dummy_type = balance_transport_problem(problem, supply, demand)
    plan_ext = vogel_method(supply_ext, demand_ext, costs_ext)
    total_cost = calculate_cost(plan_ext, costs_ext)

    end = time.perf_counter()

    return SolveResult(
        algorithm_name=algorithm_name,
        supply_modes=list(supply_modes),
        demand_modes=list(demand_modes),
        supply=supply,
        demand=demand,
        supply_ext=supply_ext,
        demand_ext=demand_ext,
        costs_ext=costs_ext,
        plan_ext=plan_ext,
        dummy_type=dummy_type,
        total_cost=total_cost,
        runtime_sec=end - start,
        extra=extra or {},
    )

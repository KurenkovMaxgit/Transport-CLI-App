from __future__ import annotations

from .models import ProblemInstance

EPS = 1e-9


def select_supply_demand(
    problem: ProblemInstance,
    supply_modes: list[int],
    demand_modes: list[int],
) -> tuple[list[float], list[float]]:
    if len(supply_modes) != problem.m:
        raise ValueError("Некоректна кількість режимів постачальників.")
    if len(demand_modes) != problem.n:
        raise ValueError("Некоректна кількість режимів споживачів.")

    supply = []
    for i, mode in enumerate(supply_modes):
        if mode < 0 or mode >= problem.k:
            raise ValueError(f"Некоректний режим потужності для постачальника {i + 1}.")
        supply.append(float(problem.A[i][mode]))

    demand = []
    for j, mode in enumerate(demand_modes):
        if mode < 0 or mode >= problem.l:
            raise ValueError(f"Некоректний режим попиту для споживача {j + 1}.")
        demand.append(float(problem.D[j][mode]))

    return supply, demand


def balance_transport_problem(
    problem: ProblemInstance,
    supply: list[float],
    demand: list[float],
) -> tuple[list[float], list[float], list[list[float]], str]:
    supply_ext = [float(x) for x in supply]
    demand_ext = [float(x) for x in demand]
    costs_ext = [[float(x) for x in row] for row in problem.C]

    total_supply = sum(supply_ext)
    total_demand = sum(demand_ext)
    diff = total_supply - total_demand

    if diff > EPS:
        demand_ext.append(diff)
        for i in range(problem.m):
            costs_ext[i].append(float(problem.over_penalty[i]))
        dummy_type = "фіктивний споживач"

    elif diff < -EPS:
        supply_ext.append(-diff)
        costs_ext.append([float(x) for x in problem.under_penalty])
        dummy_type = "фіктивний постачальник"

    else:
        dummy_type = "задача збалансована"

    return supply_ext, demand_ext, costs_ext, dummy_type

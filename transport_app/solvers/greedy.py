from __future__ import annotations

from ..evaluator import solve_for_modes
from ..models import ProblemInstance, SolveResult


def greedy_solve(problem: ProblemInstance) -> SolveResult:
  
    supply_modes = [min(range(problem.k), key=lambda p: problem.A[i][p]) for i in range(problem.m)]
    demand_modes = [min(range(problem.l), key=lambda q: problem.D[j][q]) for j in range(problem.n)]

    return solve_for_modes(
        problem,
        supply_modes,
        demand_modes,
        algorithm_name="Жадібний алгоритм",
    )

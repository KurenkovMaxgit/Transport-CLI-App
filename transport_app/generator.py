from __future__ import annotations

import random

from .models import ProblemInstance
from .validation import validate_problem


def generate_problem(
    m: int,
    n: int,
    k: int,
    l: int,
    a_mean: int,
    a_delta: int,
    d_mean: int,
    d_delta: int,
    t_mean: int,
    t_delta: int,
    penalty_extra_min: int = 5,
    penalty_extra_max: int = 30,
    seed: int | None = None,
) -> ProblemInstance:
    rng = random.Random(seed)

    a_low = max(1, a_mean - a_delta)
    a_high = max(a_low, a_mean + a_delta)

    d_low = max(1, d_mean - d_delta)
    d_high = max(d_low, d_mean + d_delta)

    t_low = max(1, t_mean - t_delta)
    t_high = max(t_low, t_mean + t_delta)

    A = []
    for _ in range(m):
        row = [rng.randint(a_low, a_high) for _ in range(k)]
        A.append(sorted(row))

    D = []
    for _ in range(n):
        row = [rng.randint(d_low, d_high) for _ in range(l)]
        D.append(sorted(row))

    C = [[rng.randint(t_low, t_high) for _ in range(n)] for _ in range(m)]

    p_low = t_high + penalty_extra_min
    p_high = t_high + penalty_extra_max

    over_penalty = [rng.randint(p_low, p_high) for _ in range(m)]
    under_penalty = [rng.randint(p_low, p_high) for _ in range(n)]

    problem = ProblemInstance(
        A=A,
        D=D,
        C=C,
        over_penalty=over_penalty,
        under_penalty=under_penalty,
    )
    validate_problem(problem)
    return problem

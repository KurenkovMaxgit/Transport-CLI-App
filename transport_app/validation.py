from __future__ import annotations

from .models import ProblemInstance


def validate_problem(problem: ProblemInstance) -> None:
    if problem.m <= 0 or problem.n <= 0:
        raise ValueError("Кількість постачальників і споживачів має бути більшою за нуль.")
    if problem.k <= 0 or problem.l <= 0:
        raise ValueError("Кількість режимів потужності й попиту має бути більшою за нуль.")

    if any(len(row) != problem.k for row in problem.A):
        raise ValueError("Матриця A має некоректну кількість стовпців.")
    if any(len(row) != problem.l for row in problem.D):
        raise ValueError("Матриця D має некоректну кількість стовпців.")
    if len(problem.C) != problem.m:
        raise ValueError("Матриця C має некоректну кількість рядків.")
    if any(len(row) != problem.n for row in problem.C):
        raise ValueError("Матриця C має некоректну кількість стовпців.")
    if len(problem.over_penalty) != problem.m:
        raise ValueError("Вектор штрафів за перевиробництво має некоректну довжину.")
    if len(problem.under_penalty) != problem.n:
        raise ValueError("Вектор штрафів за недопоставку має некоректну довжину.")

    values: list[int] = []
    for matrix in (problem.A, problem.D, problem.C):
        for row in matrix:
            values.extend(row)
    values.extend(problem.over_penalty)
    values.extend(problem.under_penalty)

    if any(x < 0 for x in values):
        raise ValueError("Усі значення задачі мають бути невід'ємними.")

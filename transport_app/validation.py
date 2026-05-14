from __future__ import annotations

from numbers import Real
from typing import Sequence

from .models import ProblemInstance


Matrix = Sequence[Sequence[Real]]
Vector = Sequence[Real]


def _validate_positive_dimensions(problem: ProblemInstance) -> None:
    if problem.m <= 0 or problem.n <= 0:
        raise ValueError("Кількість постачальників і споживачів має бути більшою за нуль.")
    if problem.k <= 0 or problem.l <= 0:
        raise ValueError("Кількість режимів потужності й попиту має бути більшою за нуль.")


def _validate_matrix_shape(
    matrix: Matrix,
    expected_rows: int,
    expected_cols: int,
    matrix_name: str,
) -> None:
    if len(matrix) != expected_rows:
        raise ValueError(
            f"Матриця {matrix_name} має некоректну кількість рядків. "
            f"Очікується {expected_rows}, отримано {len(matrix)}."
        )

    for row_index, row in enumerate(matrix, start=1):
        if len(row) != expected_cols:
            raise ValueError(
                f"Матриця {matrix_name} має некоректну кількість стовпців "
                f"у рядку {row_index}. Очікується {expected_cols}, отримано {len(row)}."
            )


def _validate_vector_length(
    vector: Vector,
    expected_len: int,
    vector_name: str,
) -> None:
    if len(vector) != expected_len:
        raise ValueError(
            f"{vector_name} має некоректну довжину. "
            f"Очікується {expected_len}, отримано {len(vector)}."
        )


def _validate_non_negative_matrix(matrix: Matrix, matrix_name: str) -> None:
    for i, row in enumerate(matrix, start=1):
        for j, value in enumerate(row, start=1):
            if not isinstance(value, Real):
                raise ValueError(
                    f"Матриця {matrix_name} містить нечислове значення "
                    f"у позиції [{i}][{j}]."
                )

            if value < 0:
                raise ValueError(
                    f"Матриця {matrix_name} містить від'ємне значення "
                    f"у позиції [{i}][{j}]."
                )


def _validate_non_negative_vector(vector: Vector, vector_name: str) -> None:
    for index, value in enumerate(vector, start=1):
        if not isinstance(value, Real):
            raise ValueError(
                f"{vector_name} містить нечислове значення у позиції {index}."
            )

        if value < 0:
            raise ValueError(
                f"{vector_name} містить від'ємне значення у позиції {index}."
            )


def validate_problem(problem: ProblemInstance) -> None:
    _validate_positive_dimensions(problem)

    _validate_matrix_shape(
        matrix=problem.A,
        expected_rows=problem.m,
        expected_cols=problem.k,
        matrix_name="A",
    )

    _validate_matrix_shape(
        matrix=problem.D,
        expected_rows=problem.n,
        expected_cols=problem.l,
        matrix_name="D",
    )

    _validate_matrix_shape(
        matrix=problem.C,
        expected_rows=problem.m,
        expected_cols=problem.n,
        matrix_name="C",
    )

    _validate_vector_length(
        vector=problem.over_penalty,
        expected_len=problem.m,
        vector_name="Вектор штрафів за перевиробництво",
    )

    _validate_vector_length(
        vector=problem.under_penalty,
        expected_len=problem.n,
        vector_name="Вектор штрафів за недопоставку",
    )

    _validate_non_negative_matrix(problem.A, "A")
    _validate_non_negative_matrix(problem.D, "D")
    _validate_non_negative_matrix(problem.C, "C")

    _validate_non_negative_vector(
        problem.over_penalty,
        "Вектор штрафів за перевиробництво",
    )

    _validate_non_negative_vector(
        problem.under_penalty,
        "Вектор штрафів за недопоставку",
    )
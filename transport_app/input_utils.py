from __future__ import annotations

from .models import ProblemInstance
from .validation import validate_problem


def read_int(prompt: str, min_value: int | None = None, default: int | None = None) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value == "" and default is not None:
            return default
        try:
            number = int(value)
            if min_value is not None and number < min_value:
                print(f"Значення має бути не менше {min_value}.")
                continue
            return number
        except ValueError:
            print("Введіть ціле число.")


def read_float(prompt: str, min_value: float | None = None, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{prompt}{suffix}: ").strip().replace(",", ".")
        if value == "" and default is not None:
            return default
        try:
            number = float(value)
            if min_value is not None and number < min_value:
                print(f"Значення має бути не менше {min_value}.")
                continue
            return number
        except ValueError:
            print("Введіть число.")


def read_int_row(expected_len: int, prompt: str) -> list[int]:
    while True:
        raw = input(prompt).strip()
        try:
            row = [int(x) for x in raw.replace(",", " ").split()]
        except ValueError:
            print("Рядок має містити тільки цілі числа.")
            continue
        if len(row) != expected_len:
            print(f"Потрібно ввести рівно {expected_len} чисел.")
            continue
        return row


def input_problem_manually() -> ProblemInstance:
    print("Введення даних задачі вручну")
    m = read_int("Кількість постачальників m", min_value=1)
    n = read_int("Кількість споживачів n", min_value=1)
    k = read_int("Кількість режимів потужності k", min_value=1)
    l = read_int("Кількість рівнів попиту l", min_value=1)

    print("\nВведіть матрицю A: можливі потужності постачальників.")
    A = [read_int_row(k, f"A[{i + 1}] ({k} чисел): ") for i in range(m)]

    print("\nВведіть матрицю D: можливі рівні попиту споживачів.")
    D = [read_int_row(l, f"D[{j + 1}] ({l} чисел): ") for j in range(n)]

    print("\nВведіть матрицю C: транспортні витрати.")
    C = [read_int_row(n, f"C[{i + 1}] ({n} чисел): ") for i in range(m)]

    print("\nВведіть штрафи за перевиробництво.")
    over_penalty = read_int_row(m, f"{m} чисел: ")

    print("\nВведіть штрафи за недопоставку.")
    under_penalty = read_int_row(n, f"{n} чисел: ")

    problem = ProblemInstance(A=A, D=D, C=C, over_penalty=over_penalty, under_penalty=under_penalty)
    validate_problem(problem)
    return problem

from __future__ import annotations

from .models import ProblemInstance, SolveResult


class Color:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(title: str) -> None:
    print("-" * 60)
    print(f"{Color.CYAN}{title}{Color.RESET}")
    print("-" * 60)


def print_matrix(title: str, matrix: list[list[float | int]]) -> None:
    print(f"\n{Color.YELLOW}{title}{Color.RESET}")
    if not matrix:
        print("<порожньо>")
        return
    for row in matrix:
        print(" ".join(f"{value:10.2f}" if isinstance(value, float) else f"{value:10d}" for value in row))


def print_vector(title: str, vector: list[float | int]) -> None:
    print(f"\n{Color.YELLOW}{title}{Color.RESET}")
    print(" ".join(f"{value:.2f}" if isinstance(value, float) else str(value) for value in vector))


def print_problem(problem: ProblemInstance) -> None:
    print_header("Дані індивідуальної задачі")
    print(f"m = {problem.m}, n = {problem.n}, k = {problem.k}, l = {problem.l}")
    print_matrix("Матриця можливих потужностей A", problem.A)
    print_matrix("Матриця можливих рівнів попиту D", problem.D)
    print_matrix("Матриця транспортних витрат C", problem.C)
    print_vector("Штрафи за перевиробництво", problem.over_penalty)
    print_vector("Штрафи за недопоставку", problem.under_penalty)


def print_solution(result: SolveResult) -> None:
    print_header(result.algorithm_name)
    print(f"Обрані режими потужності: {result.supply_modes}")
    print(f"Обрані режими попиту:     {result.demand_modes}")
    print_vector("Обрані потужності", result.supply)
    print_vector("Обраний попит", result.demand)
    print(f"\nТип балансування: {result.dummy_type}")
    print_matrix("Розширена матриця витрат", result.costs_ext)
    print_matrix("Розширений план перевезень", result.plan_ext)
    print(f"\n{Color.GREEN}Значення ЦФ: {result.total_cost:.4f}{Color.RESET}")
    print(f"Час роботи: {result.runtime_sec:.6f} с")

    if result.extra:
        if "generations" in result.extra:
            print(f"Кількість поколінь: {result.extra['generations']}")
        if "best_chromosome" in result.extra:
            print(f"Найкраща хромосома: {result.extra['best_chromosome']}")


def print_solutions_summary(results: dict[str, SolveResult]) -> None:
    print_header("Коротке порівняння розв'язків")
    if not results:
        print(f"{Color.RED}Немає розв'язків. Спочатку розв'яжіть задачу.{Color.RESET}")
        return

    for name, result in results.items():
        print(f"{name}: ЦФ = {result.total_cost:.4f}; час = {result.runtime_sec:.6f} с")

    if "greedy" in results and "genetic" in results:
        greedy = results["greedy"].total_cost
        genetic = results["genetic"].total_cost
        gap = (genetic - greedy) / greedy * 100 if greedy != 0 else 0.0
        print(f"\nВідносна різниця GA проти greedy: {gap:.2f}%")
        if genetic < greedy:
            print(f"{Color.GREEN}Генетичний алгоритм дав кращий розв'язок.{Color.RESET}")
        elif genetic > greedy:
            print(f"{Color.YELLOW}Жадібний алгоритм дав кращий розв'язок на цій задачі.{Color.RESET}")
        else:
            print("Алгоритми дали однаковий результат.")

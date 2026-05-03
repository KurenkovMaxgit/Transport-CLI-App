from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable

from .generator import generate_problem
from .input_utils import read_float, read_int
from .models import ProblemInstance
from .solvers.genetic import genetic_solve
from .solvers.greedy import greedy_solve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


def _ask_generator_params() -> dict:
    print("\nПараметри генератора задач")
    return {
        "k": read_int("Кількість режимів потужності k", 1, 4),
        "l": read_int("Кількість рівнів попиту l", 1, 4),
        "a_mean": read_int("a_сер: середнє значення потужностей", 1, 200),
        "a_delta": read_int("Δa: напівінтервал потужностей", 0, 50),
        "d_mean": read_int("d_сер: середнє значення попиту", 1, 150),
        "d_delta": read_int("Δd: напівінтервал попиту", 0, 40),
        "t_mean": read_int("t_сер: середнє значення транспортних витрат", 1, 10),
        "t_delta": read_int("Δt: напівінтервал транспортних витрат", 0, 5),
    }


def _ask_ga_params(default_max_stall_gen: int = 20) -> dict:
    print("\nПараметри генетичного алгоритму")
    return {
        "population_size": read_int("Розмір популяції", 2, 40),
        "max_stall_gen": read_int("MaxStallGen", 1, default_max_stall_gen),
        "crossover_prob": read_float("Ймовірність кросовера", 0.0, 0.8),
        "mutation_prob": read_float("Ймовірність мутації", 0.0, 0.15),
        "tournament_size": read_int("Розмір турніру", 1, 3),
    }


def _parse_float_values(raw: str, default: list[float]) -> list[float]:
    if not raw.strip():
        return default
    return [float(x.replace(",", ".")) for x in raw.split()]


def _parse_int_values(raw: str, default: list[int]) -> list[int]:
    if not raw.strip():
        return default
    return [int(x) for x in raw.split()]


def _save_csv(filename_prefix: str, rows: list[dict]) -> None:
    if not rows:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{filename_prefix}_{timestamp}.csv"

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nРезультати збережено у файл: {path}")


def _generate_task_by_size(size: int, params: dict, seed: int) -> ProblemInstance:
    return generate_problem(
        m=size,
        n=size,
        k=params["k"],
        l=params["l"],
        a_mean=params["a_mean"],
        a_delta=params["a_delta"],
        d_mean=params["d_mean"],
        d_delta=params["d_delta"],
        t_mean=params["t_mean"],
        t_delta=params["t_delta"],
        seed=seed,
    )


def _ask_dimension_experiment_params() -> tuple[int, int, int, dict, int, dict]:
    n_from = read_int("Початкова розмірність N, де m=n=N", 1, 3)
    n_to = read_int("Кінцева розмірність N", n_from, 8)
    step = read_int("Крок", 1, 1)
    generator_params = _ask_generator_params()
    r_count = read_int("Кількість задач для кожної розмірності R", 1, 5)
    ga_params = _ask_ga_params()
    return n_from, n_to, step, generator_params, r_count, ga_params


def max_stall_experiment() -> None:
    print("\nЕксперимент 1: визначення параметра умови завершення MaxStallGen")
    size = read_int("Розмірність задачі N, де m=n=N", 1, 5)
    params = _ask_generator_params()
    r_count = read_int("Кількість задач у вибірці R", 1, 5)

    population_size = read_int("Розмір популяції", 2, 40)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    mutation_prob = read_float("Ймовірність мутації", 0.0, 0.15)
    tournament_size = read_int("Розмір турніру", 1, 3)

    raw_values = input("Значення MaxStallGen через пробіл [5 10 15 20 30]: ").strip()
    stall_values = _parse_int_values(raw_values, [5, 10, 15, 20, 30])

    rows: list[dict] = []

    print("\nMaxStallGen   avg ЦФ        std ЦФ        avg generations   avg time, s")
    print("-" * 76)

    for max_stall_gen in stall_values:
        costs: list[float] = []
        times: list[float] = []
        generations: list[float] = []

        for r in range(r_count):
            problem = _generate_task_by_size(size, params, seed=1000 + r)
            result = genetic_solve(
                problem,
                population_size=population_size,
                max_stall_gen=max_stall_gen,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                tournament_size=tournament_size,
                seed=2000 + r,
            )
            costs.append(result.total_cost)
            times.append(result.runtime_sec)
            generations.append(float(result.extra.get("generations", 0)))

        row = {
            "max_stall_gen": max_stall_gen,
            "avg_cost": round(statistics.mean(costs), 6),
            "std_cost": round(statistics.pstdev(costs) if len(costs) > 1 else 0.0, 6),
            "avg_generations": round(statistics.mean(generations), 6),
            "avg_time_sec": round(statistics.mean(times), 6),
        }
        rows.append(row)
        print(
            f"{max_stall_gen:<13} "
            f"{row['avg_cost']:<13.2f} "
            f"{row['std_cost']:<13.2f} "
            f"{row['avg_generations']:<17.2f} "
            f"{row['avg_time_sec']:<13.6f}"
        )

    _save_csv("experiment_1_max_stall", rows)


def mutation_experiment() -> None:
    print("\nЕксперимент 2: вплив ймовірності мутації на ефективність ГА")
    size = read_int("Розмірність задачі N, де m=n=N", 1, 5)
    params = _ask_generator_params()
    r_count = read_int("Кількість задач у вибірці R", 1, 5)

    population_size = read_int("Розмір популяції", 2, 40)
    max_stall_gen = read_int("MaxStallGen", 1, 20)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    tournament_size = read_int("Розмір турніру", 1, 3)

    raw_values = input("Значення Pm через пробіл [0.05 0.1 0.15 0.2 0.3]: ").strip()
    mutation_values = _parse_float_values(raw_values, [0.05, 0.10, 0.15, 0.20, 0.30])

    rows: list[dict] = []

    print("\nPm        avg ЦФ        std ЦФ        avg time, s")
    print("-" * 52)

    for mutation_prob in mutation_values:
        costs: list[float] = []
        times: list[float] = []

        for r in range(r_count):
            problem = _generate_task_by_size(size, params, seed=3000 + r)
            result = genetic_solve(
                problem,
                population_size=population_size,
                max_stall_gen=max_stall_gen,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                tournament_size=tournament_size,
                seed=4000 + r,
            )
            costs.append(result.total_cost)
            times.append(result.runtime_sec)

        row = {
            "mutation_prob": mutation_prob,
            "avg_cost": round(statistics.mean(costs), 6),
            "std_cost": round(statistics.pstdev(costs) if len(costs) > 1 else 0.0, 6),
            "avg_time_sec": round(statistics.mean(times), 6),
        }
        rows.append(row)
        print(
            f"{mutation_prob:<9.2f} "
            f"{row['avg_cost']:<13.2f} "
            f"{row['std_cost']:<13.2f} "
            f"{row['avg_time_sec']:<13.6f}"
        )

    _save_csv("experiment_2_mutation", rows)


def _dimension_trials(
    metric_mode: str,
    print_header: Callable[[], None],
    print_row: Callable[[dict], None],
    filename_prefix: str,
) -> None:
    n_from, n_to, step, params, r_count, ga_params = _ask_dimension_experiment_params()
    rows: list[dict] = []

    print_header()

    for size in range(n_from, n_to + 1, step):
        greedy_costs: list[float] = []
        ga_costs: list[float] = []
        gaps: list[float] = []
        greedy_times: list[float] = []
        ga_times: list[float] = []
        ga_wins = 0

        for r in range(r_count):
            problem = _generate_task_by_size(size, params, seed=5000 + size * 100 + r)

            greedy = greedy_solve(problem)
            ga = genetic_solve(
                problem,
                population_size=ga_params["population_size"],
                max_stall_gen=ga_params["max_stall_gen"],
                crossover_prob=ga_params["crossover_prob"],
                mutation_prob=ga_params["mutation_prob"],
                tournament_size=ga_params["tournament_size"],
                seed=6000 + size * 100 + r,
            )

            greedy_costs.append(greedy.total_cost)
            ga_costs.append(ga.total_cost)
            greedy_times.append(greedy.runtime_sec)
            ga_times.append(ga.runtime_sec)

            if greedy.total_cost != 0:
                gaps.append((ga.total_cost - greedy.total_cost) / greedy.total_cost * 100)
            else:
                gaps.append(0.0)

            if ga.total_cost < greedy.total_cost:
                ga_wins += 1

        row = {
            "N": size,
            "avg_greedy_cost": round(statistics.mean(greedy_costs), 6),
            "avg_ga_cost": round(statistics.mean(ga_costs), 6),
            "avg_gap_percent": round(statistics.mean(gaps), 6),
            "ga_win_rate": round(ga_wins / r_count, 6),
            "avg_greedy_time_sec": round(statistics.mean(greedy_times), 6),
            "avg_ga_time_sec": round(statistics.mean(ga_times), 6),
        }
        rows.append(row)
        print_row(row)

    _save_csv(filename_prefix, rows)


def dimension_time_experiment() -> None:
    print("\nЕксперимент 3: вплив розмірності задачі на час роботи алгоритмів")

    def header() -> None:
        print("\nN    avg time Greedy, s   avg time GA, s")
        print("-" * 45)

    def row_printer(row: dict) -> None:
        print(
            f"{row['N']:<4} "
            f"{row['avg_greedy_time_sec']:<20.6f} "
            f"{row['avg_ga_time_sec']:<15.6f}"
        )

    _dimension_trials("time", header, row_printer, "experiment_3_dimension_time")


def dimension_accuracy_experiment() -> None:
    print("\nЕксперимент 4: вплив розмірності задачі на точність алгоритмів")

    def header() -> None:
        print("\nN    avg Greedy ЦФ   avg GA ЦФ       gap %, avg     win-rate GA")
        print("-" * 72)

    def row_printer(row: dict) -> None:
        print(
            f"{row['N']:<4} "
            f"{row['avg_greedy_cost']:<15.2f} "
            f"{row['avg_ga_cost']:<15.2f} "
            f"{row['avg_gap_percent']:<14.2f} "
            f"{row['ga_win_rate']:<13.2f}"
        )

    _dimension_trials("accuracy", header, row_printer, "experiment_4_dimension_accuracy")


def experiments_menu() -> None:
    while True:
        print("\nПроведення експериментів")
        print("1 - Визначити параметр умови завершення MaxStallGen")
        print("2 - Дослідити вплив ймовірності мутації")
        print("3 - Дослідити вплив розмірності задачі на час роботи")
        print("4 - Дослідити вплив розмірності задачі на точність")
        print("0 - Повернутися в головне меню")
        choice = input("Ваш вибір: ").strip()

        try:
            if choice == "1":
                max_stall_experiment()
            elif choice == "2":
                mutation_experiment()
            elif choice == "3":
                dimension_time_experiment()
            elif choice == "4":
                dimension_accuracy_experiment()
            elif choice == "0":
                return
            else:
                print("Невідомий пункт меню.")
        except Exception as exc:
            print(f"Помилка під час експерименту: {exc}")

from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable

from .generator import generate_problem
from .input_utils import read_float, read_int
from .models import ProblemInstance
from .plotting import save_single_series_plot, save_two_series_plot
from .solvers.genetic import (
    genetic_solve,
    estimate_max_stall_gen,
    get_alpha_max_stall,
    set_alpha_max_stall,
)
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


def _ask_alpha_for_max_stall() -> float:
    current_alpha = get_alpha_max_stall()

    alpha = read_float(
        "alpha для автоматичного MaxStallGen",
        0.01,
        current_alpha,
    )

    set_alpha_max_stall(alpha)
    print(f"Для поточного запуску програми використовується alpha = {alpha}")

    return alpha


def _print_recommended_max_stall_for_dimension(
    size: int, alpha: float | None = None
) -> int:
    if alpha is None:
        alpha = get_alpha_max_stall()

    recommended = estimate_max_stall_gen(size, size, alpha)

    print(
        f"Рекомендоване значення MaxStallGen для N={size} "
        f"при alpha={alpha}: {recommended}"
    )

    return recommended


def _make_default_stall_values(recommended: int) -> list[int]:
    values = {
        max(5, int(round(recommended * 0.25))),
        max(5, int(round(recommended * 0.50))),
        max(5, recommended),
        max(5, int(round(recommended * 1.50))),
        max(5, int(round(recommended * 2.00))),
    }

    return sorted(values)


def _ask_ga_params(
    default_max_stall_gen: int,
    allow_auto_max_stall: bool = False,
) -> dict:
    print("\nПараметри генетичного алгоритму")

    population_size = read_int("Розмір популяції", 2, 40)

    if allow_auto_max_stall:
        print(
            "Для MaxStallGen можна ввести 0, тоді значення буде автоматично "
            "рахуватися для кожної розмірності N через alpha."
        )
        max_stall_gen = read_int("MaxStallGen", 0, 0)
    else:
        max_stall_gen = read_int("MaxStallGen", 1, default_max_stall_gen)

    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    mutation_prob = read_float("Ймовірність мутації", 0.0, 0.15)
    tournament_size = read_int("Розмір турніру", 1, 3)

    return {
        "population_size": population_size,
        "max_stall_gen": max_stall_gen,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "tournament_size": tournament_size,
    }


def _parse_float_values(raw: str, default: list[float]) -> list[float]:
    if not raw.strip():
        return default
    return [float(x.replace(",", ".")) for x in raw.split()]


def _parse_int_values(raw: str, default: list[int]) -> list[int]:
    if not raw.strip():
        return default
    return [int(x) for x in raw.split()]


def _save_csv(filename_prefix: str, rows: list[dict]) -> Path | None:
    if not rows:
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{filename_prefix}_{timestamp}.csv"

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nРезультати збережено у файл: {path}")
    return path


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


def _ask_dimension_experiment_params() -> tuple[int, int, int, dict, int, dict, float]:
    n_from = read_int("Початкова розмірність N, де m=n=N", 1, 3)
    n_to = read_int("Кінцева розмірність N", n_from, 8)
    step = read_int("Крок", 1, 1)

    alpha = _ask_alpha_for_max_stall()

    print()
    _print_recommended_max_stall_for_dimension(n_from, alpha)
    if n_to != n_from:
        _print_recommended_max_stall_for_dimension(n_to, alpha)

    generator_params = _ask_generator_params()
    r_count = read_int("Кількість задач для кожної розмірності R", 1, 5)

    ga_params = _ask_ga_params(default_max_stall_gen=0, allow_auto_max_stall=True)

    return n_from, n_to, step, generator_params, r_count, ga_params, alpha


def max_stall_experiment() -> None:
    print("\nЕксперимент 1: визначення параметра умови завершення MaxStallGen")

    size = read_int("Розмірність задачі N, де m=n=N", 1, 5)
    alpha = _ask_alpha_for_max_stall()
    recommended_max_stall = _print_recommended_max_stall_for_dimension(size, alpha)

    params = _ask_generator_params()
    r_count = read_int("Кількість задач у вибірці R", 1, 5)

    population_size = read_int("Розмір популяції", 2, 40)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    mutation_prob = read_float("Ймовірність мутації", 0.0, 0.15)
    tournament_size = read_int("Розмір турніру", 1, 3)

    default_stall_values = _make_default_stall_values(recommended_max_stall)
    default_stall_text = " ".join(str(value) for value in default_stall_values)

    raw_values = input(
        f"Значення MaxStallGen через пробіл [{default_stall_text}]: "
    ).strip()
    stall_values = _parse_int_values(raw_values, default_stall_values)

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
            "N": size,
            "alpha": alpha,
            "recommended_max_stall_gen": recommended_max_stall,
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

    csv_path = _save_csv("experiment_1_max_stall", rows)

    if csv_path is not None:
        base = csv_path.with_suffix("")

        save_single_series_plot(
            x=[row["max_stall_gen"] for row in rows],
            y=[row["avg_cost"] for row in rows],
            xlabel="MaxStallGen",
            ylabel="Середнє значення ЦФ",
            title="Вплив MaxStallGen на середнє значення ЦФ",
            path=base.with_name(base.name + "_cost.png"),
        )

        save_single_series_plot(
            x=[row["max_stall_gen"] for row in rows],
            y=[row["avg_time_sec"] for row in rows],
            xlabel="MaxStallGen",
            ylabel="Середній час, с",
            title="Вплив MaxStallGen на час роботи",
            path=base.with_name(base.name + "_time.png"),
        )

        save_single_series_plot(
            x=[row["max_stall_gen"] for row in rows],
            y=[row["avg_generations"] for row in rows],
            xlabel="MaxStallGen",
            ylabel="Середня кількість поколінь",
            title="Вплив MaxStallGen на кількість поколінь",
            path=base.with_name(base.name + "_generations.png"),
        )

        print("Графіки експерименту збережено в папку output.")


def mutation_experiment() -> None:
    print("\nЕксперимент 2: вплив ймовірності мутації на ефективність ГА")

    size = read_int("Розмірність задачі N, де m=n=N", 1, 5)
    alpha = _ask_alpha_for_max_stall()
    recommended_max_stall = _print_recommended_max_stall_for_dimension(size, alpha)

    params = _ask_generator_params()
    r_count = read_int("Кількість задач у вибірці R", 1, 5)

    population_size = read_int("Розмір популяції", 2, 40)
    max_stall_gen = read_int("MaxStallGen", 1, recommended_max_stall)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    tournament_size = read_int("Розмір турніру", 1, 3)

    raw_values = input("Значення Pm через пробіл [0.05 0.1 0.15 0.2 0.3 0.4 0.5]: ").strip()
    mutation_values = _parse_float_values(raw_values, [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])

    rows: list[dict] = []

    print("\nPm        MaxStallGen   avg ЦФ        std ЦФ        avg time, s")
    print("-" * 70)

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
            "N": size,
            "alpha": alpha,
            "recommended_max_stall_gen": recommended_max_stall,
            "max_stall_gen": max_stall_gen,
            "mutation_prob": mutation_prob,
            "avg_cost": round(statistics.mean(costs), 6),
            "std_cost": round(statistics.pstdev(costs) if len(costs) > 1 else 0.0, 6),
            "avg_time_sec": round(statistics.mean(times), 6),
        }
        rows.append(row)

        print(
            f"{mutation_prob:<9.2f} "
            f"{max_stall_gen:<13} "
            f"{row['avg_cost']:<13.2f} "
            f"{row['std_cost']:<13.2f} "
            f"{row['avg_time_sec']:<13.6f}"
        )

    csv_path = _save_csv("experiment_2_mutation", rows)

    if csv_path is not None:
        base = csv_path.with_suffix("")

        save_single_series_plot(
            x=[row["mutation_prob"] for row in rows],
            y=[row["avg_cost"] for row in rows],
            xlabel="Ймовірність мутації",
            ylabel="Середнє значення ЦФ",
            title="Вплив ймовірності мутації на середнє значення ЦФ",
            path=base.with_name(base.name + "_cost.png"),
        )

        save_single_series_plot(
            x=[row["mutation_prob"] for row in rows],
            y=[row["avg_time_sec"] for row in rows],
            xlabel="Ймовірність мутації",
            ylabel="Середній час, с",
            title="Вплив ймовірності мутації на час роботи",
            path=base.with_name(base.name + "_time.png"),
        )

        print("Графіки експерименту збережено в папку output.")


def _dimension_trials(
    print_header: Callable[[], None],
    print_row: Callable[[dict], None],
    filename_prefix: str,
) -> tuple[list[dict], Path | None]:
    n_from, n_to, step, params, r_count, ga_params, alpha = (
        _ask_dimension_experiment_params()
    )
    rows: list[dict] = []

    print_header()

    for size in range(n_from, n_to + 1, step):
        greedy_costs: list[float] = []
        ga_costs: list[float] = []
        gaps: list[float] = []
        greedy_times: list[float] = []
        ga_times: list[float] = []
        ga_wins = 0

        if ga_params["max_stall_gen"] == 0:
            max_stall_gen = estimate_max_stall_gen(size, size, alpha)
        else:
            max_stall_gen = ga_params["max_stall_gen"]

        for r in range(r_count):
            problem = _generate_task_by_size(size, params, seed=5000 + size * 100 + r)

            greedy = greedy_solve(problem)
            ga = genetic_solve(
                problem,
                population_size=ga_params["population_size"],
                max_stall_gen=max_stall_gen,
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
                gaps.append(
                    (ga.total_cost - greedy.total_cost) / greedy.total_cost * 100
                )
            else:
                gaps.append(0.0)

            if ga.total_cost < greedy.total_cost:
                ga_wins += 1

        row = {
            "N": size,
            "alpha": alpha,
            "max_stall_gen": max_stall_gen,
            "avg_greedy_cost": round(statistics.mean(greedy_costs), 6),
            "avg_ga_cost": round(statistics.mean(ga_costs), 6),
            "avg_gap_percent": round(statistics.mean(gaps), 6),
            "ga_win_rate": round(ga_wins / r_count, 6),
            "avg_greedy_time_sec": round(statistics.mean(greedy_times), 6),
            "avg_ga_time_sec": round(statistics.mean(ga_times), 6),
        }
        rows.append(row)
        print_row(row)

    csv_path = _save_csv(filename_prefix, rows)
    return rows, csv_path


def dimension_time_experiment() -> None:
    print("\nЕксперимент 3: вплив розмірності задачі на час роботи алгоритмів")

    def header() -> None:
        print("\nN    alpha   MaxStallGen   avg time Greedy, s   avg time GA, s")
        print("-" * 76)

    def row_printer(row: dict) -> None:
        print(
            f"{row['N']:<4} "
            f"{row['alpha']:<7.2f} "
            f"{row['max_stall_gen']:<13} "
            f"{row['avg_greedy_time_sec']:<20.6f} "
            f"{row['avg_ga_time_sec']:<15.6f}"
        )

    rows, csv_path = _dimension_trials(
        header, row_printer, "experiment_3_dimension_time"
    )

    if csv_path is not None:
        base = csv_path.with_suffix("")

        save_two_series_plot(
            x=[row["N"] for row in rows],
            y1=[row["avg_greedy_time_sec"] for row in rows],
            y2=[row["avg_ga_time_sec"] for row in rows],
            label1="Жадібний алгоритм",
            label2="Генетичний алгоритм",
            xlabel="Розмірність задачі N",
            ylabel="Середній час, с",
            title="Вплив розмірності задачі на час роботи алгоритмів",
            path=base.with_name(base.name + "_plot.png"),
        )

        print("Графік експерименту збережено в папку output.")


def dimension_accuracy_experiment() -> None:
    print("\nЕксперимент 4: вплив розмірності задачі на точність алгоритмів")

    def header() -> None:
        print(
            "\nN    alpha   MaxStallGen   avg Greedy ЦФ   avg GA ЦФ       gap %, avg     win-rate GA"
        )
        print("-" * 98)

    def row_printer(row: dict) -> None:
        print(
            f"{row['N']:<4} "
            f"{row['alpha']:<7.2f} "
            f"{row['max_stall_gen']:<13} "
            f"{row['avg_greedy_cost']:<15.2f} "
            f"{row['avg_ga_cost']:<15.2f} "
            f"{row['avg_gap_percent']:<14.2f} "
            f"{row['ga_win_rate']:<13.2f}"
        )

    rows, csv_path = _dimension_trials(
        header, row_printer, "experiment_4_dimension_accuracy"
    )

    if csv_path is not None:
        base = csv_path.with_suffix("")

        save_two_series_plot(
            x=[row["N"] for row in rows],
            y1=[row["avg_greedy_cost"] for row in rows],
            y2=[row["avg_ga_cost"] for row in rows],
            label1="Жадібний алгоритм",
            label2="Генетичний алгоритм",
            xlabel="Розмірність задачі N",
            ylabel="Середнє значення ЦФ",
            title="Вплив розмірності задачі на значення ЦФ",
            path=base.with_name(base.name + "_cost.png"),
        )

        save_single_series_plot(
            x=[row["N"] for row in rows],
            y=[row["avg_gap_percent"] for row in rows],
            xlabel="Розмірність задачі N",
            ylabel="Середній relative gap, %",
            title="Середня відносна різниця між алгоритмами",
            path=base.with_name(base.name + "_gap.png"),
        )

        save_single_series_plot(
            x=[row["N"] for row in rows],
            y=[row["ga_win_rate"] * 100 for row in rows],
            xlabel="Розмірність задачі N",
            ylabel="Win-rate ГА, %",
            title="Частка перемог генетичного алгоритму",
            path=base.with_name(base.name + "_win_rate.png"),
        )

        print("Графіки експерименту збережено в папку output.")


def alpha_max_stall_experiment() -> None:
    print(
        "\nЕксперимент 5: дослідження коефіцієнта alpha для автоматичного MaxStallGen"
    )

    raw_dimensions = input("Значення розмірностей N через пробіл [3 5 7 10]: ").strip()
    dimensions = _parse_int_values(raw_dimensions, [3, 5, 7, 10])
    dimensions = sorted(set(value for value in dimensions if value > 0))

    if not dimensions:
        print("Список розмірностей порожній.")
        return

    raw_alpha_values = input(
        "Значення alpha через пробіл [0.1 0.15 0.2 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]: "
    ).strip()
    alpha_values = _parse_float_values(
        raw_alpha_values, [0.1, 0.15, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    alpha_values = sorted(set(value for value in alpha_values if value > 0))

    if not alpha_values:
        print("Список значень alpha порожній.")
        return

    epsilon_percent = read_float(
        "Допустиме відхилення від найкращого результату, %", 0.0, 0.0
    )

    params = _ask_generator_params()
    r_count = read_int("Кількість задач для кожної пари (N, alpha) R", 1, 5)

    print("\nПараметри генетичного алгоритму")
    population_size = read_int("Розмір популяції", 2, 40)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    mutation_prob = read_float("Ймовірність мутації", 0.0, 0.15)
    tournament_size = read_int("Розмір турніру", 1, 3)

    rows: list[dict] = []

    print("\nПочаток експерименту alpha.")

    for size in dimensions:
        print(f"\nРозмірність N = {size}")

        for alpha in alpha_values:
            max_stall_gen = estimate_max_stall_gen(size, size, alpha)
            costs: list[float] = []
            times: list[float] = []
            generations: list[float] = []

            for r in range(r_count):
                problem_seed = 7000 + size * 100 + r
                ga_seed = 8000 + size * 100 + r
                problem = _generate_task_by_size(size, params, seed=problem_seed)

                result = genetic_solve(
                    problem,
                    population_size=population_size,
                    max_stall_gen=max_stall_gen,
                    crossover_prob=crossover_prob,
                    mutation_prob=mutation_prob,
                    tournament_size=tournament_size,
                    seed=ga_seed,
                )

                costs.append(result.total_cost)
                times.append(result.runtime_sec)
                generations.append(float(result.extra.get("generations", 0)))

            row = {
                "N": size,
                "alpha": round(alpha, 6),
                "max_stall_gen": max_stall_gen,
                "avg_cost": round(statistics.mean(costs), 6),
                "std_cost": round(
                    statistics.pstdev(costs) if len(costs) > 1 else 0.0, 6
                ),
                "avg_time_sec": round(statistics.mean(times), 6),
                "avg_generations": round(statistics.mean(generations), 6),
                "gap_to_best_percent": 0.0,
                "recommended_for_N": "no",
            }
            rows.append(row)

            print(
                f"alpha={alpha:<5.2f} "
                f"MaxStallGen={max_stall_gen:<4} "
                f"avg ЦФ={row['avg_cost']:<12.2f} "
                f"avg time={row['avg_time_sec']:<10.6f}"
            )

    recommended_by_dimension: dict[int, float] = {}

    for size in dimensions:
        rows_for_size = [row for row in rows if row["N"] == size]

        if not rows_for_size:
            continue

        best_cost = min(row["avg_cost"] for row in rows_for_size)

        for row in rows_for_size:
            if best_cost != 0:
                row["gap_to_best_percent"] = round(
                    (row["avg_cost"] - best_cost) / best_cost * 100, 6
                )
            else:
                row["gap_to_best_percent"] = 0.0

        acceptable_rows = [
            row
            for row in rows_for_size
            if row["gap_to_best_percent"] <= epsilon_percent
        ]

        if acceptable_rows:
            chosen_row = min(
                acceptable_rows, key=lambda row: (row["alpha"], row["avg_time_sec"])
            )
        else:
            chosen_row = min(
                rows_for_size, key=lambda row: (row["avg_cost"], row["avg_time_sec"])
            )

        chosen_row["recommended_for_N"] = "yes"
        recommended_by_dimension[size] = chosen_row["alpha"]

    if recommended_by_dimension:
        final_alpha = max(recommended_by_dimension.values())
    else:
        final_alpha = alpha_values[0]

    print("\nПідсумкова таблиця експерименту alpha")
    print(
        "\nN    alpha   MaxStallGen   avg ЦФ        gap to best, %   avg time, s   avg generations   recommended"
    )
    print("-" * 110)

    for row in rows:
        print(
            f"{row['N']:<4} "
            f"{row['alpha']:<7.2f} "
            f"{row['max_stall_gen']:<13} "
            f"{row['avg_cost']:<13.2f} "
            f"{row['gap_to_best_percent']:<16.2f} "
            f"{row['avg_time_sec']:<13.6f} "
            f"{row['avg_generations']:<17.2f} "
            f"{row['recommended_for_N']}"
        )

    print("\nРекомендовані alpha за розмірностями:")
    for size, alpha in recommended_by_dimension.items():
        print(f"N = {size}: alpha = {alpha}")

    print(f"\nЗагальна рекомендована константа ALPHA_MAX_STALL = {final_alpha}")

    set_alpha_max_stall(final_alpha)
    print(
        f"Глобальне значення alpha для поточного запуску програми оновлено: {final_alpha}"
    )

    csv_path = _save_csv("experiment_5_alpha_max_stall", rows)

    if csv_path is not None:
        base = csv_path.with_suffix("")

        for size in dimensions:
            rows_for_size = [row for row in rows if row["N"] == size]

            save_single_series_plot(
                x=[row["alpha"] for row in rows_for_size],
                y=[row["avg_cost"] for row in rows_for_size],
                xlabel="alpha",
                ylabel="Середнє значення ЦФ",
                title=f"Вплив alpha на середнє значення ЦФ, N={size}",
                path=base.with_name(base.name + f"_N{size}_cost.png"),
            )
            save_single_series_plot(
                x=[row["alpha"] for row in rows_for_size],
                y=[row["avg_time_sec"] for row in rows_for_size],
                xlabel="alpha",
                ylabel="Середній час, с",
                title=f"Вплив alpha на час роботи ГА, N={size}",
                path=base.with_name(base.name + f"_N{size}_time.png"),
            )
            
        print("CSV та графіки експерименту alpha збережено в папку output.")


def experiments_menu() -> None:
    while True:
        print("\nПроведення експериментів")
        print("1 - Визначити параметр умови завершення MaxStallGen")
        print("2 - Дослідити вплив ймовірності мутації")
        print("3 - Дослідити вплив розмірності задачі на час роботи")
        print("4 - Дослідити вплив розмірності задачі на точність")
        print("5 - Дослідити коефіцієнт alpha для автоматичного MaxStallGen")
        print("0 - Повернутися в головне меню")
        choice = input("Ваш вибір: ").strip()

        try:
            match choice:
                case "":
                    continue
                case "1":
                    max_stall_experiment()
                case "2":
                    mutation_experiment()
                case "3":
                    dimension_time_experiment()
                case "4":
                    dimension_accuracy_experiment()
                case "5":
                    alpha_max_stall_experiment()
                case "0":
                    return
                case _:
                    print("Невідомий пункт меню.")
        except Exception as exc:
            print(f"Помилка під час експерименту: {exc}")

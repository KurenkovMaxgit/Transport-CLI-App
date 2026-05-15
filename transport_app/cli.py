from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .experiments import experiments_menu
from .formatting import (
    Color,
    print_header,
    print_problem,
    print_solution,
    print_solutions_summary,
)
from .generator import generate_problem
from .input_utils import input_problem_manually, read_float, read_int
from .io_utils import load_problem, save_problem, save_solution
from .models import ProblemInstance, SolveResult
from .solvers.genetic import genetic_solve, estimate_max_stall_gen
from .solvers.greedy import greedy_solve
from .plotting import save_ga_convergence_plot


@dataclass
class AppState:
    problem: ProblemInstance | None = None
    results: dict[str, SolveResult] = field(default_factory=dict)


def _status_line(state: AppState) -> None:
    if state.problem is None:
        print(f"{Color.RED}Немає даних.{Color.RESET}")
    else:
        print(f"{Color.GREEN}Задача задана.{Color.RESET}")


def _print_recommended_max_stall(m: int, n: int) -> int:
    recommended = estimate_max_stall_gen(m, n)

    print(
        "Рекомендоване значення MaxStallGen "
        f"(за формулою α·(m+n)·log₂(m+n)): {recommended}"
    )

    return recommended


def show_main_menu(state: AppState) -> None:
    print("-" * 60)
    print(f"{Color.BLUE}Головне меню{Color.RESET}")
    _status_line(state)
    print("1 - Введення даних задачі")
    print("2 - Розв'язати задачу всіма розробленими алгоритмами")
    print("3 - Провести експерименти")
    print("4 - Вивести дані задачі")
    print("5 - Вивести розв'язки задачі")
    print("6 - Зберегти поточну задачу у файл")
    print("7 - Зберегти розв'язки у файл")
    print("0 - Завершити роботу")
    print("-" * 60)


def input_data_menu(state: AppState) -> None:
    print_header("Введення даних задачі")
    print("Оберіть спосіб задання задачі:")
    print("1 - Ввести дані самостійно")
    print("2 - Зчитати з файлу")
    print("3 - Згенерувати випадково")
    print("0 - Повернутися в головне меню")

    choice = input("Ваш вибір: ").strip()

    match choice:
        case "":
            return

        case "1":
            state.problem = input_problem_manually()
            state.results.clear()

            print(f"{Color.GREEN}Задачу успішно введено.{Color.RESET}")
            print()
            _print_recommended_max_stall(state.problem.m, state.problem.n)

        case "2":
            path = input("Шлях до JSON-файлу [data/sample_problem.json]: ").strip()
            path = path or "data/sample_problem.json"

            state.problem = load_problem(path)
            state.results.clear()

            print(f"{Color.GREEN}Задачу успішно зчитано з файлу.{Color.RESET}")
            print()
            _print_recommended_max_stall(state.problem.m, state.problem.n)

        case "3":
            m = read_int("Кількість постачальників m", 1, 3)
            n = read_int("Кількість споживачів n", 1, 4)

            print()
            _print_recommended_max_stall(m, n)
            print(
                "Це значення буде запропоновано пізніше під час запуску генетичного алгоритму."
            )
            print()

            k = read_int("Кількість режимів потужності k", 1, 4)
            l = read_int("Кількість рівнів попиту l", 1, 4)

            a_mean = read_int("a_сер: середнє значення потужностей", 1, 200)
            a_delta = read_int("Δa: напівінтервал потужностей", 0, 50)
            d_mean = read_int("d_сер: середнє значення попиту", 1, 150)
            d_delta = read_int("Δd: напівінтервал попиту", 0, 40)
            t_mean = read_int("t_сер: середнє значення транспортних витрат", 1, 10)
            t_delta = read_int("Δt: напівінтервал транспортних витрат", 0, 5)

            seed_raw = input("Seed для генерації [можна пропустити]: ").strip()
            seed = int(seed_raw) if seed_raw else None

            state.problem = generate_problem(
                m=m,
                n=n,
                k=k,
                l=l,
                a_mean=a_mean,
                a_delta=a_delta,
                d_mean=d_mean,
                d_delta=d_delta,
                t_mean=t_mean,
                t_delta=t_delta,
                seed=seed,
            )

            state.results.clear()

            print(f"{Color.GREEN}Задачу успішно згенеровано.{Color.RESET}")
            print()
            _print_recommended_max_stall(state.problem.m, state.problem.n)

        case "0":
            return

        case _:
            print(f"{Color.RED}Невідомий пункт меню.{Color.RESET}")


def solve_all_algorithms(state: AppState) -> None:
    if state.problem is None:
        print(
            f"{Color.RED}Немає даних задачі. Спочатку введіть або згенеруйте задачу.{Color.RESET}"
        )
        return

    print_header("Розв'язання задачі всіма алгоритмами")

    print("Запускається жадібний алгоритм...")
    greedy = greedy_solve(state.problem)
    state.results["greedy"] = greedy
    print(f"Результат роботи жадібного алгоритму: значення ЦФ {greedy.total_cost:.4f}.")

    print("\nПараметри генетичного алгоритму")
    recommended_max_stall = _print_recommended_max_stall(state.problem.m, state.problem.n)
    population_size = read_int("Розмір популяції", 2, 40)
    max_stall_gen = read_int("MaxStallGen", 1, recommended_max_stall)
    crossover_prob = read_float("Ймовірність кросовера", 0.0, 0.8)
    mutation_prob = read_float("Ймовірність мутації", 0.0, 0.15)
    tournament_size = read_int("Розмір турніру", 1, 3)
    seed_raw = input("Seed для ГА [можна пропустити]: ").strip()
    seed = int(seed_raw) if seed_raw else None

    print("\nЗапускається генетичний алгоритм...")
    genetic = genetic_solve(
        state.problem,
        population_size=population_size,
        max_stall_gen=max_stall_gen,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        tournament_size=tournament_size,
        seed=seed,
    )

    state.results["genetic"] = genetic
    print(
        f"Результат роботи генетичного алгоритму: значення ЦФ {genetic.total_cost:.4f}."
    )

    history_best = genetic.extra.get("history_best", [])
    history_iteration_best = genetic.extra.get("history_iteration_best", [])

    if history_best:
        plot_path = Path("output") / "ga_convergence.png"
        save_ga_convergence_plot(
            history_best=history_best,
            history_iteration_best=history_iteration_best,
            path=plot_path,
        )
        print(f"Графік збіжності генетичного алгоритму збережено: {plot_path}")

    print_solutions_summary(state.results)


def show_problem_data(state: AppState) -> None:
    if state.problem is None:
        print(f"{Color.RED}Немає даних задачі.{Color.RESET}")
        return
    print_problem(state.problem)


def show_solutions(state: AppState) -> None:
    if not state.results:
        print(f"{Color.RED}Немає розв'язків. Спочатку розв'яжіть задачу.{Color.RESET}")
        return

    while True:
        print("\nВиведення розв'язків")
        print("1 - Коротке порівняння")
        print("2 - Жадібний алгоритм")
        print("3 - Генетичний алгоритм")
        print("0 - Повернутися")
        choice = input("Ваш вибір: ").strip()

        match choice:
            case "":
                continue
            case "1":
                print_solutions_summary(state.results)
            case "2":
                if "greedy" in state.results:
                    print_solution(state.results["greedy"])
                else:
                    print("Жадібний алгоритм ще не запускався.")
            case "3":
                if "genetic" in state.results:
                    print_solution(state.results["genetic"])
                else:
                    print("Генетичний алгоритм ще не запускався.")
            case "0":
                return
            case _:
                print("Невідомий пункт меню.")


def save_current_problem(state: AppState) -> None:
    if state.problem is None:
        print(f"{Color.RED}Немає даних задачі.{Color.RESET}")
        return
    path = (
        input("Шлях для збереження задачі [data/problem.json]: ").strip()
        or "data/problem.json"
    )
    save_problem(state.problem, path)
    print(f"{Color.GREEN}Задачу збережено: {Path(path).resolve()}{Color.RESET}")


def save_current_solutions(state: AppState) -> None:
    if not state.results:
        print(f"{Color.RED}Немає розв'язків для збереження.{Color.RESET}")
        return

    folder = Path(
        input("Папка для збереження розв'язків [data/results]: ").strip()
        or "data/results"
    )
    folder.mkdir(parents=True, exist_ok=True)

    for key, result in state.results.items():
        save_solution(result, folder / f"{key}_solution.json")

    print(f"{Color.GREEN}Розв'язки збережено в папку: {folder.resolve()}{Color.RESET}")


def run_cli() -> None:
    state = AppState()

    while True:
        show_main_menu(state)
        choice = input("Ваш вибір: ").strip()

        try:
            match choice:
                case "":
                    continue
                case "1":
                    input_data_menu(state)
                case "2":
                    solve_all_algorithms(state)
                case "3":
                    experiments_menu()
                case "4":
                    show_problem_data(state)
                case "5":
                    show_solutions(state)
                case "6":
                    save_current_problem(state)
                case "7":
                    save_current_solutions(state)
                case "0":
                    print("Роботу завершено.")
                    return
                case _:
                    print(f"{Color.RED}Невідомий пункт меню.{Color.RESET}")
        except Exception as exc:
            print(f"{Color.RED}Помилка: {exc}{Color.RESET}")

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_single_series_plot(
    x: Sequence[float | int],
    y: Sequence[float | int],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(x), list(y), marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_two_series_plot(
    x: Sequence[float | int],
    y1: Sequence[float | int],
    y2: Sequence[float | int],
    label1: str,
    label2: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(list(x), list(y1), marker="o", label=label1)
    plt.plot(list(x), list(y2), marker="o", label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_ga_convergence_plot(
    history_best: Sequence[float],
    history_iteration_best: Sequence[float] | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    generations = list(range(1, len(history_best) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(generations, list(history_best), marker="o", label="Рекордне значення ЦФ")

    if history_iteration_best and len(history_iteration_best) == len(history_best):
        plt.plot(
            generations,
            list(history_iteration_best),
            marker="o",
            label="Найкраще значення покоління",
        )

    plt.xlabel("Номер покоління")
    plt.ylabel("Значення цільової функції")
    plt.title("Динаміка роботи генетичного алгоритму")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
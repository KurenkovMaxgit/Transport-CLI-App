from __future__ import annotations

import json
from pathlib import Path

from .models import ProblemInstance, SolveResult
from .validation import validate_problem


def save_problem(problem: ProblemInstance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(problem.to_dict(), file, ensure_ascii=False, indent=2)


def load_problem(path: str | Path) -> ProblemInstance:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    problem = ProblemInstance.from_dict(data)
    validate_problem(problem)
    return problem


def save_solution(result: SolveResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)

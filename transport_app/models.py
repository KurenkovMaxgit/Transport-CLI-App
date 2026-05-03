from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ProblemInstance:
    A: list[list[int]]  
    D: list[list[int]]  
    C: list[list[int]]  
    over_penalty: list[int] 
    under_penalty: list[int]  

    @property
    def m(self) -> int:
        return len(self.A)

    @property
    def k(self) -> int:
        return len(self.A[0]) if self.A else 0

    @property
    def n(self) -> int:
        return len(self.D)

    @property
    def l(self) -> int:
        return len(self.D[0]) if self.D else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProblemInstance":
        return ProblemInstance(
            A=[[int(x) for x in row] for row in data["A"]],
            D=[[int(x) for x in row] for row in data["D"]],
            C=[[int(x) for x in row] for row in data["C"]],
            over_penalty=[int(x) for x in data["over_penalty"]],
            under_penalty=[int(x) for x in data["under_penalty"]],
        )


@dataclass
class SolveResult:
    algorithm_name: str
    supply_modes: list[int]
    demand_modes: list[int]
    supply: list[float]
    demand: list[float]
    supply_ext: list[float]
    demand_ext: list[float]
    costs_ext: list[list[float]]
    plan_ext: list[list[float]]
    dummy_type: str
    total_cost: float
    runtime_sec: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

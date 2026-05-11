from __future__ import annotations

EPS = 1e-9


def _penalty(values: list[float]) -> float:
    if not values:
        return float("-inf")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    return sorted_values[1] - sorted_values[0]


def vogel_method(
    supply: list[float],
    demand: list[float],
    costs: list[list[float]],
) -> list[list[float]]:
    supply = [float(x) for x in supply]
    demand = [float(x) for x in demand]
    costs = [[float(x) for x in row] for row in costs]

    if abs(sum(supply) - sum(demand)) > 1e-6:
        raise ValueError("Метод Фогеля отримав незбалансовану задачу.")

    m = len(supply)
    n = len(demand)
    plan = [[0.0 for _ in range(n)] for _ in range(m)]

    active_rows = [True for _ in range(m)]
    active_cols = [True for _ in range(n)]

    while any(x > EPS for x in supply) and any(x > EPS for x in demand):
        candidates: list[tuple[float, float, str, int]] = []

        for i in range(m):
            if not active_rows[i] or supply[i] <= EPS:
                continue
            cols = [j for j in range(n) if active_cols[j] and demand[j] > EPS]
            if not cols:
                continue
            row_costs = [costs[i][j] for j in cols]
            candidates.append((_penalty(row_costs), -min(row_costs), "row", i))

        for j in range(n):
            if not active_cols[j] or demand[j] <= EPS:
                continue
            rows = [i for i in range(m) if active_rows[i] and supply[i] > EPS]
            if not rows:
                continue
            col_costs = [costs[i][j] for i in rows]
            candidates.append((_penalty(col_costs), -min(col_costs), "col", j))

        if not candidates:
            break

        _, _, selected_type, selected_idx = max(candidates, key=lambda x: (x[0], x[1]))

        if selected_type == "row":
            i = selected_idx
            cols = [j for j in range(n) if active_cols[j] and demand[j] > EPS]
            j = min(cols, key=lambda col: (costs[i][col], -min(supply[i], demand[col])))
        else:
            j = selected_idx
            rows = [i for i in range(m) if active_rows[i] and supply[i] > EPS]
            i = min(rows, key=lambda row: (costs[row][j], -min(supply[row], demand[j])))

        amount = min(supply[i], demand[j])
        plan[i][j] += amount
        supply[i] -= amount
        demand[j] -= amount

        if supply[i] <= EPS:
            supply[i] = 0.0
            active_rows[i] = False
        if demand[j] <= EPS:
            demand[j] = 0.0
            active_cols[j] = False

    return plan

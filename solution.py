from queue import PriorityQueue


class Problem:
    """Transportation problem.

    Args:
        sources (int): Number of sources.
        destinations (int): Number of destinations.
        costs (list[list[int]]): Costs of transportation from source to destination.
        supply (list[int]): Supply of each source.
        demand (list[int]): Demand of each destination.
    """

    _sources: int
    """Number of sources."""
    _destinations: int
    """Number of destinations."""

    costs: dict[tuple[int, int], int]
    """Costs of transportation from source to destination."""

    supply: tuple[int, ...]
    """Supply of each source."""
    demand: tuple[int, ...]
    """Demand of each destination."""

    def __init__(
        self,
        sources: int,
        destinations: int,
        costs: list[list[int]],
        supply: list[int],
        demand: list[int],
    ):
        assert sources > 0 and destinations > 0

        self._sources = sources
        self._destinations = destinations

        assert len(costs) == sources
        assert all(len(costs[i]) == destinations for i in range(sources))

        self.costs = {}
        for i in range(sources):
            for j in range(destinations):
                assert costs[i][j] >= 0

                self.costs[i, j] = costs[i][j]

        assert len(supply) == sources
        assert len(demand) == destinations

        assert sum(supply) == sum(demand)

        self.supply = tuple(supply)
        self.demand = tuple(demand)

    def __repr__(self) -> str:
        """Serialize the problem to string.

        Returns:
            str: Serialized problem.
        """
        return f"Problem(sources={self._sources}, destinations={self._destinations}, costs={self.costs}, supply={self.supply}, demand={self.demand})"

    @property
    def sources(self) -> int:
        """Number of sources."""
        return self._sources

    @property
    def destinations(self) -> int:
        """Number of destinations."""
        return self._destinations

    def format_transportation_table(self):
        table_str = "Initial Transportation Table:\n\n"
        table_str += (
            "\t"
            + "\t".join(f"Dest{j+1}" for j in range(self._destinations))
            + "\tSupply\n"
        )
        for i in range(self._sources):
            row = "\t".join(str(self.costs[i, j]) for j in range(self._destinations))
            table_str += f"Src{i+1}\t{row}\t{self.supply[i]}\n"
        table_str += "Demand\t" + "\t".join(str(d) for d in self.demand) + "\n\n"
        return table_str


class Approximation:
    """Transportation approximation.

    Args:
        problem (Problem): Transportation problem.
    """

    _problem: Problem
    """Transportation problem reference."""

    plan: dict[tuple[int, int], int]
    """Transportation plan."""

    supply: list[int]
    """Supply of each source."""
    demand: list[int]
    """Demand of each destination."""

    def __init__(self, problem: Problem):
        self._problem = problem

        self.plan = {}

        self.supply = list(problem.supply)
        self.demand = list(problem.demand)

    def __repr__(self) -> str:
        """Serialize the approximation to string.

        Returns:
            str: Serialized approximation.
        """
        return f"Approximation(problem={self._problem}, plan={self.plan}, supply={self.supply}, demand={self.demand})"

    @property
    def problem(self) -> Problem:
        """Transportation problem."""
        return self._problem

    @property
    def sources(self) -> int:
        """Number of sources."""
        return self._problem.sources

    @property
    def destinations(self) -> int:
        """Number of destinations."""
        return self._problem.destinations

    @property
    def costs(self) -> dict[tuple[int, int], int]:
        """Costs of transportation from source to destination."""
        return self._problem.costs

    def format_result_table(self, method_name):
        table_str = f"Result Transportation Table:\n\n"
        table_str += (
            "\t"
            + "\t".join(f"Dest{j+1}" for j in range(self._problem.destinations))
            + "\tSupply\n"
        )
        for i in range(self._problem.sources):
            row = "\t".join(
                str(self.plan.get((i, j), 0)) for j in range(self._problem.destinations)
            )
            table_str += f"Src{i+1}\t{row}\t{self._problem.supply[i]}\n"
        table_str += "\n"
        return table_str


def north_west_corner(problem: Problem) -> Approximation:
    """North-West corner method.

    Args:
        problem (Problem): Transportation problem.

    Returns:
        Approximation: Transportation approximation.
    """
    approximation = Approximation(problem)

    i = 0
    j = 0

    while i < approximation.sources and j < approximation.destinations:
        approximation.plan[i, j] = min(approximation.supply[i], approximation.demand[j])

        approximation.supply[i] -= approximation.plan[i, j]
        approximation.demand[j] -= approximation.plan[i, j]

        if approximation.supply[i] == 0:
            i += 1
        else:
            j += 1

    return approximation


def penalty(approximation):
    """Calculate the penalty for each row and column.

    Args:
        approximation (Approximation): Transportation approximation.

    Returns:
        tuple[list[int | None], list[int | None]]: Penalty for each row and column.
    """
    row_penalty = [0] * approximation.sources

    for i in range(approximation.sources):
        if approximation.supply[i] == 0:
            continue

        row = [
            approximation.costs[i, j]
            for j in range(approximation.destinations)
            if (i, j) not in approximation.plan
        ]

        if len(row) >= 2:
            sorted_row = sorted(row)

            row_penalty[i] = sorted_row[1] - sorted_row[0]

    col_penalty = [0] * approximation.destinations

    for j in range(approximation.destinations):
        if approximation.demand[j] == 0:
            continue

        col = [
            approximation.costs[i, j]
            for i in range(approximation.sources)
            if (i, j) not in approximation.plan
        ]

        if len(col) >= 2:
            sorted_col = sorted(col)

            col_penalty[j] = sorted_col[1] - sorted_col[0]

    return row_penalty, col_penalty


def vogel(problem: Problem) -> Approximation:
    """Vogel's approximation method.

    Args:
        problem (Problem): Transportation problem.

    Returns:
        Approximation: Transportation approximation.
    """
    approximation = Approximation(problem)

    while max(approximation.supply) != 0 and max(approximation.demand) != 0:
        row_penalty, col_penalty = penalty(approximation)

        # print(row_penalty, col_penalty)

        max_row_penalty = max(row_penalty)
        max_col_penalty = max(col_penalty)

        if max_row_penalty >= max_col_penalty:
            i = row_penalty.index(max_row_penalty)

            min_cost = min(
                (
                    (approximation.costs[i, j], (i, j))
                    for j in range(approximation.destinations)
                    if (i, j) not in approximation.plan
                ),
                key=lambda x: x[0],
            )

            j = min_cost[1][1]
        else:
            j = col_penalty.index(max_col_penalty)

            min_cost = min(
                (
                    (approximation.costs[i, j], (i, j))
                    for i in range(approximation.sources)
                    if (i, j) not in approximation.plan
                ),
                key=lambda x: x[0],
            )

            i = min_cost[1][0]

        approximation.plan[i, j] = min(approximation.supply[i], approximation.demand[j])

        approximation.supply[i] -= approximation.plan[i, j]
        approximation.demand[j] -= approximation.plan[i, j]

    return approximation


def russell(problem: Problem) -> Approximation:
    """Russell's approximation method.

    Args:
        problem (Problem): Transportation problem.

    Returns:
        Approximation: Transportation approximation.
    """
    approximation = Approximation(problem)

    row_max = [0] * approximation.sources
    col_max = [0] * approximation.destinations

    for i in range(approximation.sources):
        for j in range(approximation.destinations):
            if approximation.costs[i, j] > row_max[i]:
                row_max[i] = approximation.costs[i, j]
            if approximation.costs[i, j] > col_max[j]:
                col_max[j] = approximation.costs[i, j]

    queue = PriorityQueue()

    for i in range(approximation.sources):
        for j in range(approximation.destinations):
            queue.put((approximation.costs[i, j] - row_max[i] - col_max[j], (i, j)))

    while not queue.empty():
        _, (i, j) = queue.get()

        approximation.plan[i, j] = min(approximation.supply[i], approximation.demand[j])

        approximation.supply[i] -= approximation.plan[i, j]
        approximation.demand[j] -= approximation.plan[i, j]

    return approximation


def calculate_total_cost(plan, costs):
    total_cost = 0
    for (i, j), transported_quantity in plan.items():
        total_cost += transported_quantity * costs[i, j]
    return total_cost

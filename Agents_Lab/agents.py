from collections import deque
from dataclasses import dataclass

WAREHOUSE = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

MOVES = (
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
)


@dataclass
class Plan:
    positions: list[tuple[int, int]]
    actions: list[str]
    states_expanded: int

    @property
    def path_length(self) -> int:
        return len(self.actions)


class WarehouseAgent:
    """Goal-based agent that plans a collision-free route with BFS."""

    def __init__(self, grid):
        if not grid or len({len(row) for row in grid}) != 1:
            raise ValueError("Grid must be non-empty and rectangular.")

        self.grid = tuple(grid)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = self._find_unique("S")
        self.goal = self._find_unique("G")

    def _find_unique(self, symbol):
        locations = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, value in enumerate(row)
            if value == symbol
        ]
        if len(locations) != 1:
            raise ValueError(f"Expected exactly one {symbol}; found {len(locations)}.")
        return locations[0]

    def successors(self, state):
        """Generate legal actions and successor states."""
        row, col = state
        for action, dr, dc in MOVES:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < self.rows
                and 0 <= nc < self.cols
                and self.grid[nr][nc] != "#"
            ):
                yield action, (nr, nc)

    def find_path(self):
        """Breadth-first search. Unit move costs make the first found path shortest."""
        frontier = deque([self.start])
        parent = {self.start: (None, None)}
        expanded = 0

        while frontier:
            state = frontier.popleft()

            if state == self.goal:
                return self._reconstruct(parent, state, expanded)

            expanded += 1
            for action, nxt in self.successors(state):
                if nxt not in parent:
                    parent[nxt] = (state, action)
                    frontier.append(nxt)

        return None

    def _reconstruct(self, parent, goal, expanded):
        positions = []
        actions = []
        state = goal

        while state is not None:
            positions.append(state)
            previous, action = parent[state]
            if action is not None:
                actions.append(action)
            state = previous

        return Plan(
            positions=list(reversed(positions)),
            actions=list(reversed(actions)),
            states_expanded=expanded,
        )

    def draw_path(self, plan):
        canvas = [list(row) for row in self.grid]
        for r, c in plan.positions[1:-1]:
            canvas[r][c] = "*"
        return "\n".join("".join(row) for row in canvas)


def validate_plan(agent, plan):
    assert plan is not None, "A path should exist for the original warehouse."
    assert plan.positions[0] == agent.start
    assert plan.positions[-1] == agent.goal
    assert len(plan.positions) == plan.path_length + 1

    for previous, current in zip(plan.positions, plan.positions[1:]):
        r1, c1 = previous
        r2, c2 = current
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert agent.grid[r2][c2] != "#"

    return True


def main():
    agent = WarehouseAgent(WAREHOUSE)
    plan = agent.find_path()
    validate_plan(agent, plan)

    print("Goal-based warehouse agent")
    print("--------------------------")
    print("Start:", agent.start)
    print("Goal :", agent.goal)
    print("Path length:", plan.path_length)
    print("States expanded:", plan.states_expanded)
    print("Actions:", " -> ".join(plan.actions))
    print("\nPath on warehouse:")
    print(agent.draw_path(plan))

    # Independent no-solution test.
    blocked = [
        "#####",
        "#S#G#",
        "#####",
    ]
    blocked_plan = WarehouseAgent(blocked).find_path()
    assert blocked_plan is None
    print("\nValidation: PASS")
    print("Blocked-grid test: PASS")


if __name__ == "__main__":
    main()

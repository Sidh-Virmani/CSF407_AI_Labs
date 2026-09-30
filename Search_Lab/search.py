from collections import deque
from dataclasses import dataclass
from itertools import count
import heapq
import math

WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

MOVES = (
    ("Up", -1, 0),
    ("Down", 1, 0),
    ("Left", 0, -1),
    ("Right", 0, 1),
)


@dataclass
class SearchResult:
    algorithm: str
    positions: list[tuple[int, int]]
    actions: list[str]
    states_expanded: int

    @property
    def found(self):
        return bool(self.positions)

    @property
    def path_length(self):
        return len(self.actions) if self.found else None


def parse_grid(grid):
    if not grid or len({len(row) for row in grid}) != 1:
        raise ValueError("Grid must be non-empty and rectangular.")

    start = goal = None
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == "S":
                if start is not None:
                    raise ValueError("Multiple start states found.")
                start = (r, c)
            elif value == "G":
                if goal is not None:
                    raise ValueError("Multiple goal states found.")
                goal = (r, c)

    if start is None or goal is None:
        raise ValueError("Grid must contain exactly one S and one G.")

    return start, goal


def successors(grid, state):
    rows, cols = len(grid), len(grid[0])
    r, c = state

    for action, dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if (
            0 <= nr < rows
            and 0 <= nc < cols
            and grid[nr][nc] != "#"
        ):
            yield action, (nr, nc)


def reconstruct(parent, goal, algorithm, expanded):
    positions, actions = [], []
    state = goal

    while state is not None:
        positions.append(state)
        previous, action = parent[state]
        if action is not None:
            actions.append(action)
        state = previous

    return SearchResult(
        algorithm=algorithm,
        positions=list(reversed(positions)),
        actions=list(reversed(actions)),
        states_expanded=expanded,
    )


def bfs(grid):
    start, goal = parse_grid(grid)
    frontier = deque([start])
    parent = {start: (None, None)}
    expanded = 0

    while frontier:
        state = frontier.popleft()

        if state == goal:
            return reconstruct(parent, goal, "BFS", expanded)

        expanded += 1
        for action, nxt in successors(grid, state):
            if nxt not in parent:
                parent[nxt] = (state, action)
                frontier.append(nxt)

    return SearchResult("BFS", [], [], expanded)


def astar(grid, heuristic, name="A*"):
    start, goal = parse_grid(grid)
    serial = count()

    # Heap entry: (f = g + h, tie breaker, g, state)
    frontier = [(heuristic(start, goal), next(serial), 0, start)]
    best_g = {start: 0}
    parent = {start: (None, None)}
    expanded = 0

    while frontier:
        _, _, g, state = heapq.heappop(frontier)

        # Ignore an outdated heap entry if a cheaper path was found later.
        if g != best_g.get(state):
            continue

        if state == goal:
            return reconstruct(parent, goal, name, expanded)

        expanded += 1
        for action, nxt in successors(grid, state):
            candidate_g = g + 1

            if candidate_g < best_g.get(nxt, math.inf):
                best_g[nxt] = candidate_g
                parent[nxt] = (state, action)
                f = candidate_g + heuristic(nxt, goal)
                heapq.heappush(frontier, (f, next(serial), candidate_g, nxt))

    return SearchResult(name, [], [], expanded)


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state, goal):
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def validate_path(grid, result):
    start, goal = parse_grid(grid)

    if not result.found:
        return False

    assert result.positions[0] == start
    assert result.positions[-1] == goal
    assert len(result.positions) == len(result.actions) + 1

    for a, b in zip(result.positions, result.positions[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert grid[b[0]][b[1]] != "#"

    return True


def run_required_tests():
    trivial = [
        "#####",
        "#SG##",
        "#####",
    ]

    no_solution = [
        "#######",
        "#S....#",
        "###.###",
        "#...#G#",
        "#######",
    ]

    alternatives = [
        "#######",
        "#S....#",
        "#.....#",
        "#....G#",
        "#######",
    ]

    original = astar(WAREHOUSE, manhattan, "A* Manhattan")
    one_step = astar(trivial, manhattan, "A* trivial")
    impossible = astar(no_solution, manhattan, "A* no solution")
    multiple = astar(alternatives, manhattan, "A* alternatives")

    assert validate_path(WAREHOUSE, original)
    assert one_step.path_length == 1 and validate_path(trivial, one_step)
    assert not impossible.found
    assert validate_path(alternatives, multiple)
    assert multiple.path_length == bfs(alternatives).path_length

    return original


def main():
    astar_result = run_required_tests()
    bfs_result = bfs(WAREHOUSE)

    assert validate_path(WAREHOUSE, bfs_result)
    assert bfs_result.path_length == astar_result.path_length

    print("Required correctness tests: PASS")
    print("\nBFS versus A*")
    print(f"{'Algorithm':<18}{'Path length':<14}{'Expanded'}")
    for result in (bfs_result, astar_result):
        print(f"{result.algorithm:<18}{result.path_length:<14}{result.states_expanded}")

    heuristic_specs = [
        ("h = 0", lambda state, goal: 0),
        ("Manhattan", manhattan),
        ("Euclidean", euclidean),
        ("2 x Manhattan", lambda state, goal: 2 * manhattan(state, goal)),
    ]

    print("\nHeuristic experiment")
    print(f"{'Heuristic':<18}{'Path length':<14}{'Expanded'}")
    for name, heuristic in heuristic_specs:
        result = astar(WAREHOUSE, heuristic, name)
        assert result.found and validate_path(WAREHOUSE, result)
        print(f"{name:<18}{result.path_length:<14}{result.states_expanded}")

    print("\nOriginal A* path:")
    print(astar_result.positions)


if __name__ == "__main__":
    main()

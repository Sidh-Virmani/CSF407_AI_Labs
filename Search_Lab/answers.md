# Search Lab — Short Notes / Answers

## Algorithms implemented

- **BFS:** blind/uninformed search using a FIFO queue.
- **A\*:** informed search using `f(n) = g(n) + h(n)`.
- Both reconstruct the actual route by storing each discovered state's parent and action.
- `states_expanded` is recorded so search effort can be compared.

## Why BFS is a valid baseline

Every move in the warehouse costs exactly one step. Therefore BFS explores nodes in nondecreasing path depth and returns a shortest path whenever one exists.

## Why Manhattan distance is suitable

For four-direction movement,

`h(n) = |row_n - row_goal| + |col_n - col_goal|`.

Ignoring obstacles can only make the estimated route shorter than or equal to the real route, so Manhattan distance is admissible. It is also consistent for unit-cost Up/Down/Left/Right movement.

## Required correctness tests

The code checks:

1. the original warehouse has a legal solution;
2. a one-step `S -> G` map returns length `1`;
3. an explicitly blocked map returns no solution;
4. a map with alternative routes returns a legal shortest path, verified against BFS.

A returned route is additionally checked to:
- start at `S`;
- end at `G`;
- move one grid square per action;
- never enter `#`.

## BFS vs A*

Both should produce an optimal path when A* uses an admissible heuristic. A* can expand fewer states when the heuristic is informative, although on some obstacle layouts the chosen heuristic may offer little advantage.

## Heuristic experiment

- **`h = 0`:** A* becomes uniform-cost search; with unit costs this behaves like BFS in terms of optimality.
- **Manhattan:** admissible and well matched to four-direction movement.
- **Euclidean:** also admissible here, but usually less informed than Manhattan for grid movement.
- **`2 × Manhattan`:** can overestimate the true remaining cost, so it is not admissible and optimality is no longer guaranteed even if it happens to find an optimal path on this map.

## Reflection

The important validation is not merely that a search function returns something. The route must satisfy the transition rules and obstacle constraints, and optimality claims should be checked against an independent baseline such as BFS.

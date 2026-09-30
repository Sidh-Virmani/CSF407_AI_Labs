# Agents Lab — Short Answers

## Task 1: Understanding the problem

1. **Environment:** a deterministic two-dimensional warehouse grid containing free cells, shelf obstacles (`#`), a start state `S`, and a goal state `G`.
2. **Goal:** move the warehouse vehicle from `S` to `G` without entering an obstacle.
3. **Actions:** move one square **Up, Down, Left, or Right** when that destination square is valid.
4. **Information maintained:** the current position, warehouse map, goal position, the search frontier, visited states, and parent/action information used to reconstruct a path.
5. **Why goal-based:** the agent chooses actions by considering which future states can lead to an explicit goal. A simple reflex agent would only map the current percept directly to an action without planning a route.

### Think About It
BFS would still be correct on a warehouse twice as large, and with unit move costs it would still find a shortest path. However, its time and especially memory use can grow substantially because it may store a large frontier. An informed method such as A* can be more efficient when a suitable heuristic is available.

## Task 2: Agent design

```text
Warehouse map + current state
            |
            v
     Successor generator
   (legal U/D/L/R moves)
            |
            v
       BFS planner
(frontier + visited + parent)
            |
            v
       Goal test: G?
            |
            v
   Reconstruct action path
```

- **State:** `(row, column)` of the vehicle.
- **Goal test:** current state equals the coordinate of `G`.
- **Decision component:** breadth-first search.
- **Transition model:** each legal action changes the position by one grid square.

## Task 3: Prompt used

> Implement a goal-based warehouse navigation agent in Python. Represent the supplied warehouse as a 2-D grid, allow only Up/Down/Left/Right moves, never cross `#`, find a collision-free path from `S` to `G`, print the path or a no-path message, and use a search strategy appropriate for equal-cost moves. Include a visited set/parent structure and tests that verify the returned path is legal.

### Questions

1. **Did the generated program work?** Yes. The final generated implementation passed both a valid-path test and an explicit blocked/no-solution test.
2. **How could the prompt be improved if it failed?** Specify the exact grid symbols, legal actions, requirement to track visited states, expected output, and correctness tests.
3. **Search algorithm chosen:** Breadth-first search (BFS).
4. **Why BFS:** every legal move has equal cost, so BFS is complete on this finite grid and the first goal path it discovers has minimum number of moves.

## Validation / reflection

The implementation was not accepted merely because it ran. The returned path is checked to begin at `S`, end at `G`, move exactly one grid cell at a time, and never enter an obstacle. A second map verifies that the program correctly reports that no path exists.

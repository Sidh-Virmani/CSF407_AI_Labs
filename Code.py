# =============================================================================
# AI LAB: LOGICAL REASONING FOR PLANNING
# Warehouse Robot Planner
#
# Core idea:
#
#           LOGIC + SEARCH = PLANNING
#
# Logic:
#   - checks whether an action is allowed in the current state
#   - calculates what the new state becomes after the action
#
# Search:
#   - explores different possible sequences of actions
#   - here we use Breadth-First Search (BFS)
#
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# dataclass lets us create a clean Action structure without manually writing
# a constructor (__init__) and several other utility functions.
from dataclasses import dataclass

# deque is used to implement the BFS queue efficiently.
#
# BFS repeatedly:
#   1. removes a state from the FRONT of the queue
#   2. generates its successor states
#   3. adds those successors to the BACK of the queue
from collections import deque


# =============================================================================
# ACTION CLASS
# =============================================================================

@dataclass(frozen=True)
class Action:
    """
    Represents ONE possible action that the robot can perform.

    Every action contains exactly the information specified in the lab:

        1. name
        2. positive preconditions
        3. negative preconditions
        4. positive effects
        5. negative effects

    Example:

        Move(A, B)

    Positive precondition:
        At(Robot, A)

    Positive effect:
        At(Robot, B)

    Negative effect:
        At(Robot, A)

    In other words:

        BEFORE:
            robot must be at A

        AFTER:
            robot is no longer at A
            robot is now at B
    """

    # Human-readable name of the action.
    # Example:
    # "Move(A,B)"
    name: str

    # Facts that MUST currently be true.
    positive_preconditions: frozenset[str]

    # Facts that MUST currently be false / absent.
    negative_preconditions: frozenset[str]

    # Facts that become true after performing the action.
    positive_effects: frozenset[str]

    # Facts that become false after performing the action.
    negative_effects: frozenset[str]


    # -------------------------------------------------------------------------
    # CHECK WHETHER ACTION IS APPLICABLE
    # -------------------------------------------------------------------------

    def is_applicable(self, state: frozenset[str]) -> bool:
        """
        Returns True if this action can legally be performed in 'state'.

        This implements the logical reasoning part:

                    S |= Preconditions(action)

        For an action to be applicable:

        1. EVERY positive precondition must exist in the current state.
        2. NONE of the negative preconditions may exist in the state.
        """

        # Example:
        #
        # positive_preconditions =
        # {
        #     "At(Robot,A)",
        #     "At(Package,A)"
        # }
        #
        # state =
        # {
        #     "At(Robot,A)",
        #     "At(Package,A)"
        # }
        #
        # issubset(state) is True because both required facts exist.

        positive_conditions_satisfied = (
            self.positive_preconditions.issubset(state)
        )

        # Negative preconditions represent things which must NOT be true.
        #
        # Example:
        #
        # negative_preconditions =
        # {
        #     "Holding(Package)"
        # }
        #
        # If "Holding(Package)" is absent from the state,
        # then the condition is satisfied.
        #
        # isdisjoint() means:
        # "these two sets have no common elements."

        negative_conditions_satisfied = (
            self.negative_preconditions.isdisjoint(state)
        )

        # BOTH requirements must hold.
        return (
            positive_conditions_satisfied
            and negative_conditions_satisfied
        )


    # -------------------------------------------------------------------------
    # APPLY AN ACTION TO A STATE
    # -------------------------------------------------------------------------

    def apply(self, state: frozenset[str]) -> frozenset[str]:
        """
        Performs this action and returns the successor state.

        According to the lab specification:

        1. remove negative effects
        2. add positive effects

        Mathematically:

            S' = (S - NegativeEffects) U PositiveEffects
        """

        # Safety check:
        # We should never apply an action whose preconditions are false.
        if not self.is_applicable(state):
            raise ValueError(
                f"Action {self.name} cannot be applied to this state."
            )

        # STEP 1:
        # Remove facts which become false.
        #
        # Example:
        #
        # state:
        #     At(Robot,A)
        #
        # Move(A,B) negative effect:
        #     At(Robot,A)
        #
        # therefore it is removed.

        state_after_removing_negative_effects = (
            state - self.negative_effects
        )

        # STEP 2:
        # Add facts which become true.
        #
        # Example:
        #
        # Move(A,B) positive effect:
        #     At(Robot,B)

        new_state = (
            state_after_removing_negative_effects
            | self.positive_effects
        )

        return new_state


# =============================================================================
# SMALL HELPER FUNCTION FOR CREATING ACTIONS
# =============================================================================

def make_action(
    name,
    positive_preconditions=(),
    negative_preconditions=(),
    positive_effects=(),
    negative_effects=()
):
    """
    Convenience function.

    Instead of repeatedly writing:

        frozenset({...})

    for every field, this function converts the supplied sets automatically.

    It does NOT perform planning.
    It only makes creating Action objects cleaner.
    """

    return Action(
        name=name,
        positive_preconditions=frozenset(positive_preconditions),
        negative_preconditions=frozenset(negative_preconditions),
        positive_effects=frozenset(positive_effects),
        negative_effects=frozenset(negative_effects)
    )


# =============================================================================
# BREADTH-FIRST SEARCH PLANNER
# =============================================================================

def bfs_planner(initial_state, goal, actions):
    """
    Finds a sequence of actions that transforms the initial state
    into a state satisfying the goal.

    BFS is used because the lab specifically asks for Breadth-First Search.

    BFS explores plans according to their LENGTH.

    It first checks:
        plans of length 0

    then:
        plans of length 1

    then:
        plans of length 2

    then:
        plans of length 3

    and so on.

    Therefore, assuming every action has equal cost,
    the first plan BFS finds uses the minimum number of actions.

    ---------------------------------------------------------
    PARAMETERS
    ---------------------------------------------------------

    initial_state:
        Set of facts true at the beginning.

    goal:
        Set of facts that must eventually become true.

    actions:
        List of all actions available to the robot.

    ---------------------------------------------------------
    RETURNS
    ---------------------------------------------------------

    If a plan exists:

        plan, state_history

    Otherwise:

        None, None
    """

    # Convert the initial state to frozenset.
    #
    # Why frozenset instead of normal set?
    #
    # Normal sets can be modified and cannot be stored inside another set.
    #
    # frozenset is immutable and hashable, meaning we can store states inside
    # the "visited" set used by BFS.
    initial_state = frozenset(initial_state)

    # Same idea for the goal.
    goal = frozenset(goal)


    # -------------------------------------------------------------------------
    # BFS QUEUE
    # -------------------------------------------------------------------------
    #
    # Every element in the queue stores THREE things:
    #
    #   1. current state
    #   2. actions used to reach this state
    #   3. complete history of states reached
    #
    # Initially:
    #
    # current state = initial state
    # plan          = empty
    # history       = [initial state]

    queue = deque([
        (
            initial_state,       # current state
            [],                  # no actions performed yet
            [initial_state]      # state history begins at S0
        )
    ])


    # -------------------------------------------------------------------------
    # VISITED SET
    # -------------------------------------------------------------------------
    #
    # The search might otherwise repeatedly return to the same state.
    #
    # Example:
    #
    #       A -> B -> A -> B -> A -> ...
    #
    # visited prevents this infinite repetition.
    visited = {initial_state}


    # -------------------------------------------------------------------------
    # MAIN BFS LOOP
    # -------------------------------------------------------------------------

    while queue:

        # Remove the OLDEST state from the queue.
        #
        # This FIFO behaviour is what makes the algorithm Breadth-First Search.
        current_state, current_plan, state_history = queue.popleft()


        # ---------------------------------------------------------------------
        # GOAL TEST
        # ---------------------------------------------------------------------
        #
        # Goal:
        #
        #     At(Package,C)
        #
        # We check whether every required goal fact exists in the state.
        #
        # IMPORTANT:
        # We do NOT check whether Robot is at C.
        #
        # The actual goal from the lab is PACKAGE at C.

        if goal.issubset(current_state):

            # Goal reached.
            # Return the successful plan and its state transitions.
            return current_plan, state_history


        # ---------------------------------------------------------------------
        # TRY EVERY AVAILABLE ACTION
        # ---------------------------------------------------------------------

        for action in actions:

            # LOGICAL REASONING:
            #
            # Is this action actually possible in the current state?
            if action.is_applicable(current_state):

                # If yes, apply its effects and generate a successor state.
                successor_state = action.apply(current_state)


                # -------------------------------------------------------------
                # AVOID REPEATED STATES
                # -------------------------------------------------------------

                if successor_state not in visited:

                    # Record that we have now discovered this state.
                    visited.add(successor_state)


                    # ---------------------------------------------------------
                    # BUILD THE NEW PLAN
                    # ---------------------------------------------------------
                    #
                    # Suppose current plan was:
                    #
                    #     PickUp(Package,A)
                    #     Move(A,B)
                    #
                    # and the new action is:
                    #
                    #     Move(B,C)
                    #
                    # then new_plan becomes:
                    #
                    #     PickUp(Package,A)
                    #     Move(A,B)
                    #     Move(B,C)

                    new_plan = current_plan + [action]


                    # Also keep the sequence of actual states.
                    new_history = state_history + [successor_state]


                    # Add this unexplored successor to the back of the BFS queue.
                    queue.append(
                        (
                            successor_state,
                            new_plan,
                            new_history
                        )
                    )


    # -------------------------------------------------------------------------
    # BFS FINISHED WITHOUT REACHING GOAL
    # -------------------------------------------------------------------------
    #
    # If queue becomes empty, then every reachable state has been examined.
    #
    # Therefore no valid plan exists.

    return None, None


# =============================================================================
# PRINTING FUNCTIONS
# =============================================================================

def print_state(state):
    """
    Prints all propositions in a state in a readable way.
    """

    # sorted() is only for nicer deterministic display.
    print("{ " + ", ".join(sorted(state)) + " }")


def print_result(plan, state_history):
    """
    Prints the final plan together with every state transition.
    """

    # No plan exists.
    if plan is None:
        print("No plan found")
        return


    print("\nPLAN FOUND")
    print("=" * 60)


    # S0 is always the initial state.
    print("\nS0:")
    print_state(state_history[0])


    # enumerate(..., start=1) gives:
    #
    # 1, first action
    # 2, second action
    # ...
    #
    # state_history[i] is the state AFTER action i.

    for i, action in enumerate(plan, start=1):

        print(f"\nAction {i}: {action.name}")

        print(f"S{i}:")
        print_state(state_history[i])


    print("\nFinal plan:")

    for i, action in enumerate(plan, start=1):
        print(f"{i}. {action.name}")


# =============================================================================
# DEFINE THE WAREHOUSE PROBLEM
# =============================================================================

def create_warehouse_actions():
    """
    Creates the actions available in the warehouse problem.

    Warehouse locations:

        A <--> B <--> C

    Actions:

        Move
        PickUp
        Drop

    The actions below are "grounded", meaning that instead of having a general
    variable-based action Move(X,Y), we explicitly construct:

        Move(A,B)
        Move(B,A)
        Move(B,C)
        Move(C,B)

    This keeps the implementation simple and matches the lab problem.
    """

    actions = []


    # =========================================================================
    # MOVEMENT ACTIONS
    # =========================================================================

    # -------------------------------------------------------------------------
    # Move(A,B)
    # -------------------------------------------------------------------------
    #
    # Preconditions:
    #
    #     At(Robot,A)
    #
    # Effects:
    #
    #     NOT At(Robot,A)
    #     At(Robot,B)

    actions.append(
        make_action(
            name="Move(A,B)",

            positive_preconditions={
                "At(Robot,A)"
            },

            negative_preconditions=set(),

            positive_effects={
                "At(Robot,B)"
            },

            negative_effects={
                "At(Robot,A)"
            }
        )
    )


    # -------------------------------------------------------------------------
    # Move(B,A)
    # -------------------------------------------------------------------------

    actions.append(
        make_action(
            name="Move(B,A)",

            positive_preconditions={
                "At(Robot,B)"
            },

            negative_preconditions=set(),

            positive_effects={
                "At(Robot,A)"
            },

            negative_effects={
                "At(Robot,B)"
            }
        )
    )


    # -------------------------------------------------------------------------
    # Move(B,C)
    # -------------------------------------------------------------------------

    actions.append(
        make_action(
            name="Move(B,C)",

            positive_preconditions={
                "At(Robot,B)"
            },

            negative_preconditions=set(),

            positive_effects={
                "At(Robot,C)"
            },

            negative_effects={
                "At(Robot,B)"
            }
        )
    )


    # -------------------------------------------------------------------------
    # Move(C,B)
    # -------------------------------------------------------------------------

    actions.append(
        make_action(
            name="Move(C,B)",

            positive_preconditions={
                "At(Robot,C)"
            },

            negative_preconditions=set(),

            positive_effects={
                "At(Robot,B)"
            },

            negative_effects={
                "At(Robot,C)"
            }
        )
    )


    # =========================================================================
    # PICKUP AND DROP ACTIONS
    # =========================================================================
    #
    # The robot may pick up or drop the package at A, B, or C.
    #
    # We use a loop instead of manually writing six nearly identical actions.

    for location in ["A", "B", "C"]:

        # ---------------------------------------------------------------------
        # PickUp(Package, location)
        # ---------------------------------------------------------------------
        #
        # Example for location A:
        #
        # Preconditions:
        #
        #     At(Robot,A)
        #     At(Package,A)
        #
        # Negative precondition:
        #
        #     NOT Holding(Package)
        #
        # Effects:
        #
        #     Holding(Package)
        #     NOT At(Package,A)

        actions.append(
            make_action(
                name=f"PickUp(Package,{location})",

                positive_preconditions={
                    f"At(Robot,{location})",
                    f"At(Package,{location})"
                },

                # The robot should not already be holding the package.
                negative_preconditions={
                    "Holding(Package)"
                },

                positive_effects={
                    "Holding(Package)"
                },

                negative_effects={
                    f"At(Package,{location})"
                }
            )
        )


        # ---------------------------------------------------------------------
        # Drop(Package, location)
        # ---------------------------------------------------------------------
        #
        # Example for C:
        #
        # Preconditions:
        #
        #     At(Robot,C)
        #     Holding(Package)
        #
        # Effects:
        #
        #     At(Package,C)
        #     NOT Holding(Package)

        actions.append(
            make_action(
                name=f"Drop(Package,{location})",

                positive_preconditions={
                    f"At(Robot,{location})",
                    "Holding(Package)"
                },

                negative_preconditions=set(),

                positive_effects={
                    f"At(Package,{location})"
                },

                negative_effects={
                    "Holding(Package)"
                }
            )
        )


    return actions


# =============================================================================
# TEST A: ORIGINAL SOLVABLE WAREHOUSE PROBLEM
# =============================================================================

def test_a():
    """
    Test A from the lab:

    Original warehouse problem.

    Initial state:
        Robot at A
        Package at A

    Goal:
        Package at C

    A valid plan should be found.
    """

    print("\n")
    print("=" * 70)
    print("TEST A: SOLVABLE WAREHOUSE PROBLEM")
    print("=" * 70)


    # Initial state I from the lab.
    initial_state = {
        "At(Robot,A)",
        "At(Package,A)"
    }


    # Goal G from the lab.
    goal = {
        "At(Package,C)"
    }


    # Generate all warehouse actions.
    actions = create_warehouse_actions()


    # Ask BFS to search for a plan.
    plan, history = bfs_planner(
        initial_state,
        goal,
        actions
    )


    # Display results.
    print_result(plan, history)


# =============================================================================
# TEST B: IMPOSSIBLE PROBLEM
# =============================================================================

def test_b():
    """
    Test B from the lab:

    Remove PickUp actions.

    The robot therefore has no way to begin carrying the package.

    Even though the robot itself can reach C,
    At(Package,C) can NEVER become true.

    Expected result:

        No plan found
    """

    print("\n")
    print("=" * 70)
    print("TEST B: IMPOSSIBLE PROBLEM")
    print("=" * 70)


    initial_state = {
        "At(Robot,A)",
        "At(Package,A)"
    }


    goal = {
        "At(Package,C)"
    }


    # Start with normal warehouse actions.
    actions = create_warehouse_actions()


    # Remove EVERY PickUp action.
    #
    # action.name.startswith("PickUp") identifies all:
    #
    # PickUp(Package,A)
    # PickUp(Package,B)
    # PickUp(Package,C)

    actions_without_pickup = [
        action
        for action in actions
        if not action.name.startswith("PickUp")
    ]


    plan, history = bfs_planner(
        initial_state,
        goal,
        actions_without_pickup
    )


    print_result(plan, history)


# =============================================================================
# TEST C: IRRELEVANT ROBOT MOVEMENT
# =============================================================================

def test_c():
    """
    Test C checks an important logical distinction:

        At(Robot,C)

    is NOT the same proposition as:

        At(Package,C)

    Merely moving the robot to C does not automatically complete the goal.

    We add an extra action allowing the robot to move directly from A to C.

    The planner must still check the ACTUAL goal:
        At(Package,C)
    """

    print("\n")
    print("=" * 70)
    print("TEST C: IRRELEVANT ROBOT MOVEMENT")
    print("=" * 70)


    initial_state = {
        "At(Robot,A)",
        "At(Package,A)"
    }


    goal = {
        "At(Package,C)"
    }


    actions = create_warehouse_actions()


    # Add an extra movement action.
    #
    # Notice carefully:
    #
    # It changes:
    #     At(Robot,A) -> At(Robot,C)
    #
    # It does NOT change:
    #     At(Package,A)
    #
    # Therefore simply executing this action does NOT satisfy the goal.

    actions.append(
        make_action(
            name="DirectMove(A,C)",

            positive_preconditions={
                "At(Robot,A)"
            },

            negative_preconditions=set(),

            positive_effects={
                "At(Robot,C)"
            },

            negative_effects={
                "At(Robot,A)"
            }
        )
    )


    plan, history = bfs_planner(
        initial_state,
        goal,
        actions
    )


    print_result(plan, history)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":
    """
    Python begins executing the program here.

    We run all three tests requested in the lab:

        Test A -> solvable
        Test B -> impossible
        Test C -> irrelevant movement

    Keeping the tests separate also makes it easier to demonstrate to the TA
    that the planner has actually been validated instead of being run only once.
    """

    test_a()
    test_b()
    test_c()
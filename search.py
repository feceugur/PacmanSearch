# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    "*** YOUR CODE HERE ***"
    # initializations
    visited = set()  # Keeps track of visited nodes and their directions
    solution = []  # Stores the sequence of directions for Pacman to reach the goal state
    stack = util.Stack() # Initialize Pacman's backpack (a stack)
    parents = {}  # Dictionary to maintain parent-child relationships

    # Pacman starts his quest
    start = problem.getStartState()
    stack.push((start, 'Unknown', 0))
    visited.add(start)

    if problem.isGoalState(start):
        return solution

    while not stack.isEmpty():
        # pop from top of stack
        node, direction, cost = stack.pop()
        # store element and its direction
        visited.add(node)
        # check if element is goal
        if problem.isGoalState(node):
            node_sol = node
            break
        # expand node
        for successor, succ_direction, succ_cost in problem.getSuccessors(node):
            # If Pacman hasn't been here before
            if successor not in visited:
                # Pacman makes a new friend (family connection)
                parents[successor] = node
                # Pacman adds the new place to his backpack
                stack.push((successor, succ_direction, succ_cost))

    # Reconstruct Pacman solution path by backtracking through parents
    while node_sol in parents:
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # initializations
    visited = {}  # Keeps track of visited nodes and their directions
    solution = []  # Stores the sequence of directions for Pacman to reach the goal state
    queue = util.Queue()  # Queue for BFS exploration
    parents = {}  # Dictionary to maintain parent-child relationships

    # Get the start state and enqueue it
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solution

    # loop while queue is not empty and goal is not reached
    while not queue.isEmpty():
        # Dequeue from the front of the queue
        node = queue.pop()
        visited[node[0]] = node[1]

        # Check if the current node is the goal state
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            break

        for successor, succ_direction, succ_cost in problem.getSuccessors(node[0]):
            # Check if the successor has not been visited or expanded as a child of another node
            if successor not in visited.keys() and successor not in parents.keys():
                # Maintain the parent-child relationship
                parents[successor] = node[0]
                # Enqueue the successor for exploration
                queue.push((successor, succ_direction, succ_cost))

    # finding and storing the path
    while node_sol in parents.keys():
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution


def uniformCostSearch(problem):
    """Search the node of the least total cost first."""
    "*** YOUR CODE HERE ***"


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initializations
    visited = {}  # Keeps track of visited nodes and their directions
    solution = []  # Stores the sequence of directions for Pacman to reach the goal state
    queue = util.PriorityQueue()  # Priority queue for A* exploration
    parents = {}  # Dictionary to maintain parent-child relationships
    cost = {}  # Dictionary to maintain costs for nodes

    # Get the start state and enqueue it
    start = problem.getStartState()
    priority = 0 + heuristic(start, problem)
    queue.push((start, 'Undefined', 0), priority)
    visited[start] = 'Undefined'
    cost[start] = 0

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solution

    # Perform A* search until the queue is empty or the goal is reached
    while not queue.isEmpty():
        # Dequeue from the front of the priority queue
        node = queue.pop()
        visited[node[0]] = node[1]

        # Check if the current node is the goal state
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            break
        # expand node
        for successor, succ_direction, succ_cost in problem.getSuccessors(node[0]):
            # If the successor is not visited, calculate its new cost
            if successor not in visited.keys():
                new_cost = node[2] + succ_cost
                priority = new_cost + heuristic(successor, problem)

                # If cost of successor was calculated earlier while expanding a different node,
                # if the new cost is more than the old cost, continue
                if successor in cost.keys() and cost[successor] <= priority:
                    continue

                # If the new cost is less than the old cost, push to the priority queue and update cost and parent
                queue.push((successor, succ_direction, new_cost), priority)
                cost[successor] = new_cost
                parents[successor] = node[0]

    # finding and storing the path
    while node_sol in parents.keys():
        # find parent
        node_sol_prev = parents[node_sol]
        # prepend direction to solution
        solution.insert(0, visited[node_sol])
        # go to previous node
        node_sol = node_sol_prev

    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

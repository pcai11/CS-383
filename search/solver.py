import collections
import sys
import puzz
import pdqpq


MAX_SEARCH_ITERS = 100000
GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, strategy):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state: an EightPuzzleBoard object indicating the start state for the search
        flavor: a string indicating which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
            'path' - a list of 2-tuples representing the path from the start state to the goal state 
                (both should be included), with each entry being a (str, EightPuzzleBoard) pair 
                indicating the move and resulting state for each action.  Omitted if the search 
                fails.
            'path_cost' - the total cost of the path, taking into account the costs associated 
                with each state transition.  Omitted if the search fails.
            'frontier_count' - the number of unique states added to the search frontier at any
                point during the search.
            'expanded_count' - the number of unique states removed from the frontier and expanded 
                (i.e. have successors generated).
    """
    path = []
    path_cost = 0
    frontier_count = 1
    expanded_count = 0

    frontier = pdqpq.PriorityQueue()
    explored = set()
    start = ("start", start_state)

    if(start_state == GOAL_STATE):
        return{
            'path': (start, start),
            'path_cost': path_cost,
            'frontier_count': frontier_count,
            'expanded_count': expanded_count
        }

    if strategy == 'bfs':
        history = dict()
        history.update({start_state: (start_state, start, 0)})
        frontier.add(start)
        found = False
        while not frontier.empty():
            if found == True:
                break
            node = frontier.pop()
            explored.add(node[1])
            expanded_count += 1
            states = expand(node)
            for move in states:
                child = states[move]
                cost = find_cost(node, child)
                if(child not in frontier) and (child not in explored):
                    history.update({child: (node[1], (move, child), cost + parent_cost(history, node[1]))})
                    if is_goal(child):
                        node = (move, child)
                        found = True
                        break
                    else:
                        frontier_count += 1
                        frontier.add((move, child))
        path_cost = history[node[1]][2]
        path = reconstruct(start, node[1], history)
    
    
    elif strategy == 'ucost':
        history = dict()
        history.update({start_state: (start_state, (start), 0)})
        frontier.add(start, 0)
        while not frontier.empty():
            node = frontier.pop()
            explored.add(node[1])
            states = expand(node)
            expanded_count += 1
            if is_goal(node[1]):
                break
            for n in states:
                cost = find_cost(node, states[n])
                cumulative_cost = parent_cost(history, node[1])
                if(states[n] not in frontier) and (states[n] not in explored):
                    frontier_count += 1
                    frontier.add((n, states[n]), cost + cumulative_cost)
                    history.update({states[n]: (node[1], (n, states[n]), cost + cumulative_cost)})
                elif(states[n] in frontier) and (frontier.get(states[n]) > cost + cumulative_cost):
                    frontier.add((n, states[n]), cost + cumulative_cost)
                    history.update({states[n]: (node[1], (n, states[n]), cost + cumulative_cost)})
        path_cost = history[node[1]][2]
        path = reconstruct(start, node[1], history)


    elif strategy == 'greedy-h1' or strategy == 'greedy-h2' or strategy == 'greedy-h3':
        return greedy_search(start, strategy)

    elif strategy == 'astar-h1' or strategy == 'astar-h2' or strategy == 'astar-h3':
        return astar_search(start, strategy)

    results = {
        'path': path,
        'path_cost': path_cost,
        'frontier_count': frontier_count,
        'expanded_count': expanded_count
    }
    if not path:
        results.pop('path')
    return results


def astar_search(start, strategy):
    if strategy == 'astar-h1':
        h = misplaced_tiles
    elif strategy == 'astar-h2':
        h = manhattan_distance
    elif strategy == 'astar-h3':
        h = hybrid_heuristic
    start_state = start[1]
    path = []
    path_cost = 0
    frontier_count = 1
    expanded_count = 0
    frontier = pdqpq.PriorityQueue()
    explored = set()
    history = dict()
    history.update({start_state: (start_state, start, 0)})
    heuristic = h(start)
    frontier.add(start, heuristic)
    while not frontier.empty():
        node = frontier.pop()
        explored.add(node[1])
        states = expand(node)
        expanded_count += 1
        if is_goal(node[1]):
            break
        for n in states:
            cost = find_cost(node, states[n])
            heuristic = h((n, states[n]))
            cumulative_cost = parent_cost(history, node[1])
            if(states[n] not in frontier) and (states[n] not in explored):
                frontier_count += 1
                frontier.add((n, states[n]), heuristic + cumulative_cost)
                history.update({states[n]: (node[1], (n, states[n]), cost + cumulative_cost)})
            elif(states[n] in frontier) and (frontier.get(states[n]) > heuristic + cumulative_cost):
                frontier.add((n, states[n]), heuristic + cumulative_cost)
                history.update({states[n]: (node[1], (n, states[n]), cost + cumulative_cost)})
                
    path_cost = history[node[1]][2]
    path = reconstruct(start, node[1], history)
    results = {
        'path': path,
        'path_cost': path_cost,
        'frontier_count': frontier_count,
        'expanded_count': expanded_count
    }
    if not path:
        results.pop('path')
    return results


def greedy_search(start, strategy):
    if strategy == 'greedy-h1':
        h = misplaced_tiles
    elif strategy == 'greedy-h2':
        h = manhattan_distance
    elif strategy == 'greedy-h3':
        h = hybrid_heuristic
    start_state = start[1]
    path = []
    path_cost = 0
    frontier_count = 1
    expanded_count = 0
    frontier = pdqpq.PriorityQueue()
    explored = set()
    history = dict()
    history.update({start_state: (start_state, start, 0)})
    heuristic = h(start)
    frontier.add(start, heuristic)
    while not frontier.empty():
        node = frontier.pop()
        if is_goal(node[1]):
            break
        explored.add(node[1])
        states = expand(node)
        expanded_count += 1
        for n in states:
            cost = find_cost(node, states[n])
            heuristic = h((n, states[n]))
            if(states[n] not in frontier) and (states[n] not in explored):
                frontier_count += 1
                frontier.add((n, states[n]), heuristic)
                history.update({states[n]: (node[1], (n, states[n]), cost + parent_cost(history, node[1]))})
            elif(states[n] in frontier) and (frontier.get(states[n]) > heuristic):
                frontier.add((n, states[n]), cost)
                history.update({states[n]: (node[1], (n, states[n]), cost)})
    path_cost = history[node[1]][2]
    path = reconstruct(start, node[1], history)
    results = {
        'path': path,
        'path_cost': path_cost,
        'frontier_count': frontier_count,
        'expanded_count': expanded_count
    }
    if not path:
        results.pop('path')
    return results


def parent_cost(history, board):
    return history[board][2]


def hybrid_heuristic(current):
    count = 0
    for i in range(1,9):
        x, y = current[1].find(str(i))
        goal_x, goal_y = GOAL_STATE.find(str(i))
        if(x != goal_x or y != goal_y):
            count += i ** 2 * (abs(x - goal_x) + abs(y - goal_y))
    return count


def manhattan_distance(current):
    count = 0
    for i in range(1,9):
        x, y = current[1].find(str(i))
        goal_x, goal_y = GOAL_STATE.find(str(i))
        count += abs(x - goal_x) + abs(y - goal_y)
    return count


def misplaced_tiles(current):
    count = 0
    for i in range(1,9):
        x, y = current[1].find(str(i))
        goal_x, goal_y = GOAL_STATE.find(str(i))
        if(x != goal_x or y != goal_y):
            count += 1
    return count
    

def reconstruct(start, end, history):
    path = []
    move = history[end][1]
    path.insert(0, move)
    parent = history[end][0]
    while start not in path:
        move = history[parent][1]
        path.insert(0, move)
        parent = history[parent][0]
    return path


def find_cost(current, state):
    x, y = current[1].find('0')
    cost = int(state._get_tile(x, y)) ** 2
    return cost


def expand(node): 
    return node[1].successors()


def is_goal(node):
    return GOAL_STATE == node


def print_summary(results):
    if 'path' in results:
        print("found solution of length {}, cost {}".format(len(results['path']), 
                                                            results['path_cost']))
        for move, state in results['path']:
            print("  {:5} {}".format(move, state))
    else:
        print("no solution found")
    print("{} states placed on frontier, {} states expanded".format(results['frontier_count'], 
                                                                    results['expanded_count']))


############################################

if __name__ == '__main__':

    start = puzz.EightPuzzleBoard(sys.argv[1])
    method = sys.argv[2]
    """
    state = puzz.EightPuzzleBoard("123470568")
    print(state)
    print(state.pretty())
    succs = state.successors()
    print(succs)
    d = succs['down']
    print(d.pretty())
    """

    print("solving puzzle {} -> {}".format(start, GOAL_STATE))
    results = solve_puzzle(start, method)
    print_summary(results)

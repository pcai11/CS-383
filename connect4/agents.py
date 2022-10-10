import random
import math


BOT_NAME = "Jouster Fei" #+ 19 


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        nextp = state.next_player()
        best_score = -math.inf if nextp == 1 else math.inf
        successor_states = state.successors()

        if not successor_states and state.is_full():
            return state.utility()
        
        else:
            for move, state in successor_states:
                score = self.minimax(state)
                if ((nextp == 1) and (score > best_score)) or ((nextp == -1) and (score < best_score)):
                    best_score = score

        return best_score 

class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.minimax_helper(state, self.depth_limit)

        
    def minimax_helper(self, state, depth):
        nextp = state.next_player()
        best_score = -math.inf if nextp == 1 else math.inf
        successor_states = state.successors()

        if depth <= 0 or (not successor_states and state.is_full()):
            return self.evaluation(state)

        else:
            for move, state in successor_states:
                score = self.minimax_helper(state, depth - 1)
                if ((nextp == 1) and (score > best_score)) or ((nextp == -1) and (score < best_score)):
                    best_score = score

        return best_score 


    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        score = 0

        rows = state.get_rows()
        cols = state.get_cols()
        diags = state.get_diags()

        num_streaks_rows = self.eval_helper(rows)
        num_streaks_cols = self.eval_helper(cols)
        num_streaks_diags = self.eval_helper(diags)

        score = self.score_helper(num_streaks_rows) + self.score_helper(num_streaks_cols) + self.score_helper(num_streaks_diags)
        
        return score


    def score_helper(self, streaks):
        sum_scores = 0
        curr_score = 0
        
        for id in streaks:
            streak = streaks[id]
            curr_score = streak[0]**2 * streak[1]
            if streak[0] < 0:
                curr_score = -curr_score
        sum_scores += curr_score
            
        
        return sum_scores
            

    def eval_helper(self, structure):
        streaks = {}
        current_streak = 1
        start_of_streak = 0
        curr_count = 1
        streak_id = 0
        prev = structure[0][0]

        for i in structure:

            if curr_count >= len(i):
                continue
            
            curr = i[curr_count]
            curr_count += 1

            if prev == curr:
                if prev == -1:
                    current_streak = -current_streak
                current_streak += 1
                streaks.update({streak_id: (current_streak, 1)})

            elif curr == 0:
                if start_of_streak > 0 and current_streak > 1:
                    if i[start_of_streak-1] == 0:
                        streaks[streak_id] = (streaks[streak_id][0], 2)

            else:
                start_of_streak = curr_count - 1
                streak_id += 1
                current_streak = 1

            prev = curr
        
        return streaks

class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        successor_states = state.successors()
        nextp = state.next_player()
        best_score = -math.inf if nextp == 1 else math.inf
        alpha = -math.inf
        beta = math.inf

        if not successor_states and state.is_full():
            return 0

        for move, state in successor_states:

            if nextp == 1:
               score = self.minimize_beta(state, alpha, beta)
            else:
                score = self.maximize_alpha(state, alpha, beta)

            if nextp == 1:
                if score > best_score:
                    best_score = score
            else:
                if score < best_score:
                    best_score = score
        
        return best_score


    def minimize_beta(self, state, a, b):
        successor_states = state.successors()
        beta = b
        
        if not successor_states and state.is_full():
            return state.utility()
        
        else:
            for move, state in successor_states:
                if a < beta:
                    score = self.maximize_alpha(state, a, beta)
                
                if score < beta:
                    beta = score
        
        return beta


    
    def maximize_alpha(self, state, a, b):
        successor_states = state.successors()
        alpha = a

        if not successor_states and state.is_full():
            return state.utility()

        else:
            for move, state in successor_states:
                if alpha < b:
                    score = self.minimize_beta(state, alpha, b)
                
                if score > alpha:
                    alpha = score
        
        return alpha


# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.


from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from game_runner import run_many
import pickle
import torch
from torch import nn


MAXIMIZER = 0
MIMIZER = 1

class GameAgent():
    # Interface for Game agents
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        # Given a state and time limit, return an action
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)

        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)

# ------------------------------------------------------------------------------
#  MINIMAX
# ------------------------------------------------------------------------------

# def minimax(asp: GoProblem, state: GameState, time_limit: float) -> Action:
def minimax(asp: GoProblem, state: GameState, cutoff_depth: int) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        state - (GameState) current game state
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.

        ** time_limit - (float) time limit for agent to return a move

    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))

    """

    # Helper function for finding max value
    # def max_value_helper(state: GameState, start_time: float, time_limit: float) -> tuple[Action, int]:
    def max_value_helper(state: GameState, depth: int) -> tuple[Action, int]:

        if  asp.is_terminal_state(state): 
            return (None, asp.evaluate_terminal(state))
        
        # elif (time.time - start_time) >= time_limit: 
        #     return (None, asp.heuristic(state, 0))
        elif depth >= cutoff_depth: 
            return (None, asp.heuristic(state, state.player_to_move()))
        
        else: 

            # Initialize max value to negative infinity
            best_value = float('-inf')
            best_actions = []
            
            # Iterate through possible actions and next states 
            for action in asp.get_available_actions(state): 
                
                # Get next state of the action
                next_state = asp.transition(state, action)
                # Recursive action - will store max value of the possible next states
                # (_, value) =  min_value_helper(next_state, start_time, time_limit)
                (_, value) =  min_value_helper(next_state, depth + 1)
                if value > best_value:
                    best_value = value
                    # best_action = action
                    best_actions.clear()
                    best_actions.append(action)
                elif value == best_value:
                    best_actions.append(action)

            # Return max value
            return (random.choice(best_actions), best_value)

    # Helper function for finding min value  
    # def min_value_helper(state: GameState, start_time: float, time_limit: float) -> tuple[Action, int]:
    def min_value_helper(state: GameState, depth: int) -> tuple[Action, int]:

        if  asp.is_terminal_state(state): 
            return (None, asp.evaluate_terminal(state))
        
        # elif (time.time - start_time) >= time_limit: 
        #     return (None, asp.heuristic(state, 1))
        elif depth >= cutoff_depth: 
            return (None, asp.heuristic(state, state.player_to_move()))
        
        else:

            # Initialize max value to negative infinity
            # best_action = (None, float('inf'))

            best_value = float('inf')
            best_actions = []

            # Iterate through possible actions and next states 
            for action in asp.get_available_actions(state): 
                
                # Get next state of the action
                next_state = asp.transition(state, action)
                # Recursive action - will store min value of the possible next states
                (_, value) = max_value_helper(next_state, depth + 1)
                if value < best_value:
                    best_value = value
                    # best_action = action
                    best_actions.clear()
                    best_actions.append(action)
                elif value == best_value:
                    best_actions.append(action)
            # Return min value
            return (random.choice(best_actions), best_value)

    # ------- START -------
    # Initialize best action and stats
    # best_action = None
    # stats = {
    #     'states_expanded': 0
    # }

    # Get start time
    start_time = time.time()

    # Get start state
    start_state = state
    
    # Determine which player's move it is
    current_player = start_state.player_to_move()

    # Current Player is Maximizer
    if current_player == 0: 
   
        # Get Best Actin
        # (action, _) = max_value_helper(start_state, start_time, time_limit)
        (action, _) = max_value_helper(start_state, 1)

        return action
    
    # Current Player is Minimizer
    else: 
        
        # Get Best Action
        # (action, _) = min_value_helper(start_state, start_time, time_limit)  
        (action, _) = min_value_helper(start_state, 1)  

        return action
    
class MinimaxAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move method of MinimaxAgent
        return minimax(self.search_problem, game_state,  cutoff_depth=self.depth)


    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)

# ------------------------------------------------------------------------------
#  ALPHA BETA
# ------------------------------------------------------------------------------

# def alpha_beta(asp: GoProblem, state: GameState, time_limit: float) -> Action:
def alpha_beta(asp: GoProblem, state: GameState, cutoff_depth: int) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        state - (GameState) current game state
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.

        time_limit - (float) time limit for agent to return a move
        
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    
    # Helper function for finding max value
    # def max_value_helper(state: GameState, start_time: float, time_limit: float, alpha: int, beta: int) -> tuple[Action, int, int, int]:
    def max_value_helper(state: GameState, depth: int, alpha: int, beta: int) -> tuple[Action, int, int, int]:
        if  asp.is_terminal_state(state): 
            return (None, asp.evaluate_terminal(state), alpha, beta)
        
        # elif (time.time - start_time) >= time_limit: 
        #     return (None, asp.heuristic(state, 0), alpha, beta)
        elif depth >= cutoff_depth: 
            return (None, asp.heuristic(state, 0), alpha, beta)
        
        else: 

            # Initialize max value to negative infinity
            # best_action = (None, float('-inf'))
            best_value = float('-inf')
            best_actions = []

            # Iterate through possible actions and next states 
            for action in asp.get_available_actions(state): 
                
                # If best value is greater than beta --> prune
                #  - Comparing best value for the maximizer to best option for the minimizer (beta) higher in tree
                #  - If value is greater than beta, then maximizer can prune since it knows minimzer will never
                #    pick the action with the lower value 
                if best_value > beta: 
                    # prune
                    break 
                else: 
                    # Get next state of the action
                    next_state = asp.transition(state, action)
                    # Recursive action - will store max value of the possible next states
                    # (_, value, alpha_return, beta_return) =  min_value_helper(next_state, start_time, time_limit, alpha, beta)
                    (_, value, alpha_return, beta_return) =  min_value_helper(next_state, depth + 1, alpha, beta)
                    
                    # Update best value for the maximizer 
                    if value > best_value:
                        best_value = value
                    # best_action = action
                        best_actions.clear()
                        best_actions.append(action)
                    elif value == best_value:
                        best_actions.append(action)
            

            # Return max value
            return (random.choice(best_actions), best_value, alpha, beta)

    # Helper function for finding min value  
    # def min_value_helper(state: GameState, start_time: float, time_limit: float, alpha: int, beta: int) -> tuple[Action, int, int, int]:
    def min_value_helper(state: GameState, depth: int, alpha: int, beta: int) -> tuple[Action, int, int, int]:
        if  asp.is_terminal_state(state): 
            return (None, asp.evaluate_terminal(state), alpha, beta)
        
        # elif (time.time - start_time) >= time_limit: 
        #     return (None, asp.heuristic(state, 1), alpha, beta)
        elif depth >= cutoff_depth: 
            return (None, asp.heuristic(state, 1), alpha, beta)
        else:

            # Initialize max value to negative infinity
            # best_action = (None, float('inf'))
            best_value = float('inf')
            best_actions = []

            # Iterate through possible actions and next states 
            for action in asp.get_available_actions(state): 
                
                # If best value is less than alpha --> prune
                # - Comparing best value for minimizer to the best option for the maximizer (alpha) higher in tree
                # - If the value is less than alpha, then the minizer can prune since it knows the maximizer will
                #   never pick the action with the lower value 
                if best_value < alpha: 
                    # prune
                    break 
                else: 
                    # Get next state of the action
                    next_state = asp.transition(state, action)
                    # Recursive action - will store min value of the possible next states
                    # (_, value, alpha_return, beta_return) = max_value_helper(next_state, start_time, time_limit, alpha, beta)
                    (_, value, alpha_return, beta_return) = max_value_helper(next_state, depth + 1, alpha, beta)

                    if value < best_value:
                        best_value = value
                        # best_action = action
                        best_actions.clear()
                        best_actions.append(action)
                    elif value == best_value:
                        best_actions.append(action)

            # Return min value
            return (random.choice(best_actions), best_value, alpha, beta)

    # ------- START -------
    # Initialize best action and stats
    # best_action = None
    # stats = {
    #     'states_expanded': 0
    # }

    # Get start time
    start_time = time.time()
    
    # Get start state
    start_state = state
    
    # Determine which player's move it is
    current_player = start_state.player_to_move()

    # Initialize alpha and beta
    # - alpha = best already explored option for maximizer
    alpha = float('-inf') 
    # - beta = best already explored option for minimzer
    beta = float('inf')

    # Current Player is Maximizer
    if current_player == 0: 
   
        # Get Best Actin
        # (action, value, alpha_return, beta_return) = max_value_helper(start_state, start_time, time_limit, alpha, beta)
        (action, value, alpha_return, beta_return) = max_value_helper(start_state, 1, alpha, beta)

        return (action, value)
    
    # Current Player is Minimizer
    else: 
        
        # Get Best Action
        # (action, value, alpha_return, beta_return) = min_value_helper(start_state, start_time, time_limit, alpha, beta) 
        (action, value, alpha_return, beta_return) = min_value_helper(start_state, 1, alpha, beta) 

        return (action, value)

class AlphaBetaAgent(GameAgent):
    def __init__(self, depth=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of AlphaBeta Agent
        (action, _) = alpha_beta(self.search_problem, game_state,  cutoff_depth=self.depth)
        return action 

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: implement get_move algorithm of IterativeDeepeningAgent
        
        # Get start time 
        start_time = time.time()

        # Determine which player's move it is
        current_player = game_state.player_to_move()

        # Current Player is Maximizer
        if current_player == 0: 
            best_value = float('-inf')
            best_action = None

            cutoff_depth = 1
            current_duration = time.time() - start_time
            while current_duration < time_limit:
                (action, value) = alpha_beta(self.search_problem, game_state,  cutoff_depth=cutoff_depth)
                
                if value > best_value:
                    best_value = value
                    best_action = action

                cutoff_depth += 1
                current_duration = time.time() - start_time
            
            return best_action
            
        # Current Player is Minimizer
        else: 
            best_value = float('inf')
            best_action = None

            cutoff_depth = 1
            current_duration = time.time() - start_time
            while current_duration < time_limit:
                (action, value) = alpha_beta(self.search_problem, game_state,  cutoff_depth=cutoff_depth)
            
                if value < best_value:
                    best_value = value
                    best_action = action

                cutoff_depth += 1
                current_duration = time.time() - start_time
            
            return best_action

    def __str__(self):
        return f"IterativeDeepneing + " + str(self.search_problem)


class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)


class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c

        # Initialize Search problem
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using MCTS algorithm
        
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        # TODO: Implement MCTS
        pass

    def __str__(self):
        return "MCTS"

def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
      super(ValueNetwork, self).__init__()

      # TODO: What should the output size of a Value function be?
      output_size = 1

      # TODO: Add more layers, non-linear functions, etc.=
      self.linear = nn.Linear(input_size, output_size)
      
      # self.fc1 = nn.Linear(input_size, 128)
      # self.fc2 = nn.Linear(128, 64)
      # self.fc3 = nn.Linear(64, 32)
      # self.fc4 = nn.Linear(32, 16)
      # self.fc5 = nn.Linear(16, output_size)
      
      self.fc1 = nn.Linear(input_size, 32)
      self.fc2 = nn.Linear(32, 16)
      self.fc3 = nn.Linear(16, 8)
      self.fc4 = nn.Linear(8, output_size)

      self.sigmoid = nn.Sigmoid()
      self.tanh = nn.Tanh()
      self.relu = nn.ReLU() 
      


    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # TODO: Update as more layers are added

      z1 = self.fc1(x)
      a1 = self.relu(z1)
      
      z2 = self.fc2(a1)
      a2 = self.tanh(z2)
      
      z3 = self.fc3(a2)
      a3 = self.relu(z3)

      z4 = self.fc4(a3)
      # a4 = self.tanh(z4)
      output = self.sigmoid(z4)

      # z5 = self.fc5(a4)
      # output = self.sigmoid(z5)

      return output

class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model
        
    def __call__(self, model=None):
        """
        Use the model to compute a heuristic value for a given state.
        """
        return self

    def encoding(self, state):
        """
        Get encoding of state (convert state to features)
        Note, this may call get_features() from Task 1. 

        Input:
            state: GoState to encode into a fixed size list of features
        Output:
            features: list of features
        """
        # TODO: get encoding of state (convert state to features)
        features = get_features(state)

        return features

    def heuristic(self, state, player_index):
        """
        Return heuristic (value) of current state

        Input:
            state: GoState to encode into a fixed size list of features
            player_index: index of player to evaluate heuristic for
        Output:
            value: heuristic (value) of current state
        """
        # TODO: Compute heuristic (value) of current state
        features = self.encoding(state)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        value = self.model(features_tensor)
        
        # if player_index == 1:
        #     value = -value

        # Note, your agent may perform better if you force it not to pass
        # (i.e., don't select action #25 on a 5x5 board unless necessary)
        return value

    def __str__(self) -> str:
        return "Learned Heuristic"


def create_value_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
    """

    model_path = "value_model.pt"
    # TODO: Update number of features for your own encoding size
    feature_size = 100 #UPDATE THIS NUMBER WITH THE NUMBER FROM YOUR NOTEBOOK
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    learned_agent = GreedyAgent(heuristic_search_problem) # MAKE SURE THIS IS GREEDY

    return learned_agent


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, board_size=5):
      super(PolicyNetwork, self).__init__()

      # TODO: What should the output size of the Policy be?
      output_size = (board_size * board_size) + 1

      # TODO: Add more layers, non-linear functions, etc.
      self.linear = nn.Linear(input_size, output_size)
      
      self.fc1 = nn.Linear(input_size, 32)
      self.fc2 = nn.Linear(32, 16)
      self.fc3 = nn.Linear(16, 8)
      self.fc4 = nn.Linear(8, output_size)

      self.sigmoid = nn.Sigmoid()
      self.tanh = nn.Tanh()
      self.relu = nn.ReLU() 
      self.softmax = nn.Softmax(dim=-1) 
      # needed for class probabilities; output probabilities sum to 1

    def forward(self, x):
      # TODO: Update as more layers are added

      z1 = self.fc1(x)
      a1 = self.relu(z1)
      
      z2 = self.fc2(a1)
      a2 = self.relu(z2)
      
      z3 = self.fc3(a2)
      a3 = self.relu(z3)

      z4 = self.fc4(a3)
      output = self.sigmoid(z4)

      return output
    
def get_features(game_state: GoState):
    """
    Map a game state to a list of features.

    Some useful functions from game_state include:
        game_state.size: size of the board
        get_pieces_coordinates(player_index): get coordinates of all pieces of a player (0 or 1)
        get_pieces_array(player_index): get a 2D array of pieces of a player (0 or 1)
        
        get_board(): get a 2D array of the board with 4 channels (player 0, player 1, empty, and player to move). 4 channels means the array will be of size 4 x n x n
    
        Descriptions of these methods can be found in the GoState

    Input:
        game_state: GoState to encode into a fixed size list of features
    Output:
        features: list of features
    """
    board_size = game_state.size
    # black_pieces = game_state.get_pieces_array(0).flatten()
    # white_pieces = game_state.get_pieces_array(1).flatten()

    # One-hot encoding of stone locations (n * n * 2 features)
    # stone_locations = np.concatenate((black_pieces, white_pieces)).tolist()

    # Player to move ( 0 = black, 1 = white)
    # player_to_move = game_state.player_to_move()


    # TODO: Encode game_state into a list of features
    features = [] 
    # features.extend(board)
    # features.append(player_to_move)
    features = np.array(game_state.get_board()).flatten()
    return features

class PolicyAgent(GameAgent):
    def __init__(self, search_problem, model_path, board_size=5):
        super().__init__()
        self.search_problem = search_problem
        # self.model = load_model(model_path, PolicyNetwork)
        self.board_size = board_size

        feature_size = board_size ** 2 * 4
        self.model = load_model(model_path, PolicyNetwork(feature_size, board_size))
        self.model.eval()

    def encoding(self, state):
        # TODO: get encoding of state (convert state to features)
        features = get_features(state)

        return features

    def get_move(self, game_state, time_limit=1):
      """
      Get best action for current state using self.model

      Input:
        game_state: current state of the game
        time_limit: time limit for search (This won't be used in this agent)
      Output:
        action: best action to take
      """

      # TODO: Select LEGAL Best Action predicted by model

      features = self.encoding(game_state)
      features_tensor = torch.tensor(features, dtype=torch.float32)
      
      probabilities = self.model(features_tensor)
      legal_actions = self.search_problem.get_available_actions(game_state)

      # The top prediction of your model may not be a legal move!
      best_action = random.choice(legal_actions)
      action_probabilities = {}

      for action in legal_actions:
          action_probabilities[action] = probabilities[action]

      best_prob = float('-inf') 

      for action, prob in action_probabilities.items(): 
          if prob > best_prob: 
              best_prob = prob
              best_action = action

      if best_action == 25: 
          action_probabilities[best_action] -= .2
      
      for action, prob in action_probabilities.items(): 
          if prob > best_prob: 
              best_prob = prob
              best_action = action

      # Note, you may want to force your policy not to pass their turn unless necessary
      assert best_action in self.search_problem.get_available_actions(game_state)
      
      return best_action

    def __str__(self) -> str:
        return "Policy Agent"
    
def main():
    agent1 = GreedyAgent()
    agent2 = GreedyAgent()
    # Play 10 games
    run_many(agent1, agent2, 10)


if __name__ == "__main__":
    main()

import numpy as np
import torch
from torch import nn
# from game_runner import run_many
from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
from abc import ABC, abstractmethod
import random
import time
# from agents import MCTSNode

# ------------------------------------------------------------------------------
#  PREVIOUS PART CODE
# ------------------------------------------------------------------------------

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

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()

        # TODO: What should the output size of a Value function be?
        output_size = 1

        # TODO: Add more layers, non-linear functions, etc.=
        self.linear = nn.Linear(input_size, output_size)
      
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
        output = self.sigmoid(z4)

        return output
      

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
   
    features = [] 
 
    features = np.array(game_state.get_board()).flatten()

    return features

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

# ------------------------------------------------------------------------------
#  FINAL AGENT CODE
# ------------------------------------------------------------------------------

# def alpha_beta(asp: GoProblem, state: GameState, time_limit: float) -> Action:
def alpha_beta_ids(asp: GoProblem, state: GameState, cutoff_depth: int, start_time: float, cutoff_time: float) -> Action:
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
        elif (time.time() - start_time) >= cutoff_time:
            return (None, asp.heuristic(state, 0), alpha, beta)
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
                    (_, value, _, _) =  min_value_helper(next_state, depth + 1, alpha, beta)
                    
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
        elif (time.time() - start_time) >= cutoff_time:
            return (None, asp.heuristic(state, 1), alpha, beta)
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
                    (_, value, _, _) = max_value_helper(next_state, depth + 1, alpha, beta)

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

    # Get start time
    current_time = time.time()
        
    while (current_time - start_time) < cutoff_time:
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
            (action, value, _, _) = max_value_helper(start_state, 1, alpha, beta)

            return (action, value)
        
        # Current Player is Minimizer
        else: 
            
            # Get Best Action
            # (action, value, alpha_return, beta_return) = min_value_helper(start_state, start_time, time_limit, alpha, beta) 
            (action, value, _, _) = min_value_helper(start_state, 1, alpha, beta) 

            return (action, value)
        
    return None
    
class FinalAgent(GameAgent):
    def __init__(self, cutoff_time=0.9, model_path="value_model.pt", input_size=4*5*5):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.move_counter = 0
        
        # Load ValueNetwork
        self.value_network = ValueNetwork(input_size)
        self.value_network = load_model(model_path, self.value_network)
        self.value_network.eval()  # Set to evaluation mode

        # Set Value Network as Search Problem Heuristic 
        self.search_problem = GoProblem(5)  # Assuming board size is 5 x 5
        self.search_problem.heuristic = self.value_network_heuristic

        # Opening Book
        self.opening_moves = [12, 6, 8, 16, 18]

    def value_network_heuristic(self, state, player):
        """
        Heuristic function that uses ValueNetwork to evaluate the game state.

        Args:
            state (GoState): current game state
            player (int): player (0 or 1) to move
        Returns:
            heuristic value (float): ValueNetwork output for the state
        """
        features = torch.tensor(get_features(state), dtype=torch.float32).unsqueeze(0)

        output = self.value_network(features).item()

        # Adjust heuristic to align with current player
        return output if player == 0 else -output

    def get_move(self, game_state, time_limit):
        """
        Iterative deepening alpha-beta search with ValueNetwork as heuristic.
        """
        start_time = time.time()

        if len(self.search_problem.get_available_actions(game_state)) >= 24:
            self.move_counter = 0
        
        self.move_counter += 1
        print("Move ", self.move_counter)
    
        legal_actions = self.search_problem.get_available_actions(game_state)
        
        if self.move_counter <= 3:
            for action in self.opening_moves: 
                if action in legal_actions:
                    print("Using opening book move...")
                    return action
            
        best_action = random.choice(legal_actions)
        current_depth = 2
        max_depth = float('inf')
            
        while (time.time() - start_time) < self.cutoff_time and current_depth <= max_depth:
            action, _ = alpha_beta_ids(
                self.search_problem, game_state, cutoff_depth=current_depth, 
                start_time=start_time, cutoff_time=self.cutoff_time
            )
            if (time.time() - start_time) > self.cutoff_time:
                break
            if action is not None:
                best_action = action
            current_depth += 1

        return best_action

    def __str__(self):
        return "FinalAgent with ValueNetwork"


def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    try: return FinalAgent()
    except: raise NotImplementedError


def main():
    from game_runner import run_many

    agent1 = get_final_agent_5x5()
    agent2 = GreedyAgent()
    # Play 10 games
    run_many(agent1, agent2, 10)


if __name__ == "__main__":
    main() 
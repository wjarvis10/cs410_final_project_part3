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

        #Prior Probaility 
        self.prior_prob = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)

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
        # a4 = self.tanh(z4)
        output = self.sigmoid(z4)

        # z5 = self.fc5(a4)
        # output = self.sigmoid(z5)

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

class FinalAgent:
    def __init__(self, value_model_path="value_model.pt", policy_model_path="policy_model.pt", mcts_c=np.sqrt(2), board_size=5):
        """
        Initializes the FinalAgent, combining MCTS, Value Network, and Policy Network.

        Args:
            value_model (nn.Module): Pretrained Value Network model.
            policy_model (nn.Module): Pretrained Policy Network model.
            mcts_c (float): Exploration constant for MCTS.
            board_size (int): Size of the Go board.
        """
        feature_size = 100
        self.value_model = load_model(value_model_path, ValueNetwork(feature_size))
        self.policy_model = load_model(policy_model_path, PolicyNetwork(feature_size, board_size))
        self.mcts_c = mcts_c
        self.board_size = board_size
        self.search_problem = GoProblem()
        self.move_counter = 0

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Gets the best move for the current game state using MCTS guided by Value and Policy Networks.

        Args:
            game_state (GoState): The current game state.
            time_limit (float): Time limit for the decision.

        Returns:
            Action: The best action to take.
        """
        root_node = MCTSNode(state=game_state)
        start_time = time.time()

        print("Move ", self.move_counter)
        self.move_counter += 1
        
        time_limit = 1 #Update this for better uses of time

        legal_moves = root_node.state.legal_actions()

        best_value = 0
        best_action = random.choice(legal_moves)

        while time.time() - start_time < time_limit:
            leaf_node = self._select(root_node)
            self._expand(leaf_node)
            simulation_result = self._simulate(leaf_node)
            self._backpropagate(leaf_node, simulation_result)
            best_child = max(root_node.children, key=lambda child: child.visits)
            if best_child.visits > best_value:
                best_value = best_child.visits
                best_action = best_child.action

        # Choose the action leading to the most visited child node
        # best_action = max(root_node.children, key=lambda child: child.visits).action

        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a node to expand using PUCT (Policy-based UCT).

        Args:
            node (MCTSNode): The current node.

        Returns:
            MCTSNode: The selected node.
        """
        while node.children:
            best_node = max(
                node.children,
                key=lambda child: (child.value / child.visits if child.visits > 0 else float('inf')) +
                                self.mcts_c * child.prior_prob * np.sqrt(node.visits / (child.visits + 1))
            )
            node = best_node
        return node


    def _expand(self, node: MCTSNode):
        """
        Expands the node by adding all possible children.

        Args:
            node (MCTSNode): The node to expand.
        """
        if self.search_problem.is_terminal_state(node.state):
            return

        actions = self.search_problem.get_available_actions(node.state)
        state_features = self._encode_state(node.state)
        with torch.no_grad():
            policy_output = self.policy_model(torch.tensor(state_features, dtype=torch.float32))

        for i, action in enumerate(actions):
            next_state = self.search_problem.transition(node.state, action)
            prior_prob = policy_output[i].item()
            child_node = MCTSNode(state=next_state, parent=node, action=action)
            child_node.prior_prob = prior_prob
            node.children.append(child_node)

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulates a game from the node to the end using the Value Network.

        Args:
            node (MCTSNode): The node to simulate from.

        Returns:
            float: The value of the end state from the perspective of the current player.
        """
        features = self._encode_state(node.state)
        with torch.no_grad():
            value = self.value_model(torch.tensor(features, dtype=torch.float32)).item()
        return value

    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Backpropagates the result of a simulation up the tree.

        Args:
            node (MCTSNode): The node where backpropagation starts.
            result (float): The result of the simulation.
        """
        while node is not None:
            node.visits += 1
            if node.state.player_to_move() == 0:  # Maximizer
                node.value += result
            else:  # Minimizer
                node.value -= result
            node = node.parent

    def _encode_state(self, state: GoState):
        """
        Encodes a game state into features for the neural networks.

        Args:
            state (GoState): The game state to encode.

        Returns:
            np.ndarray: Encoded features.
        """
        return np.array(state.get_board()).flatten()

    def __str__(self):
        return "Final Agent"

def main():
    from game_runner import run_many

    agent1 = FinalAgent()
    agent2 = GreedyAgent()
    # Play 10 games
    run_many(agent1, agent2, 10)


if __name__ == "__main__":
    main()
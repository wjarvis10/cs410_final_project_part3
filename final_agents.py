import numpy as np
import torch
from torch import nn
from game_runner import run_many
from go_search_problem import GoProblem, GoState, Action
from heuristic_go_problems import *
from abc import ABC, abstractmethod
import random
import time

class FinalAgent:
    def __init__(self, value_model, policy_model, mcts_c=np.sqrt(2), board_size=5):
        """
        Initializes the FinalAgent, combining MCTS, Value Network, and Policy Network.

        Args:
            value_model (nn.Module): Pretrained Value Network model.
            policy_model (nn.Module): Pretrained Policy Network model.
            mcts_c (float): Exploration constant for MCTS.
            board_size (int): Size of the Go board.
        """
        self.value_model = value_model
        self.policy_model = policy_model
        self.mcts_c = mcts_c
        self.board_size = board_size
        self.search_problem = GoProblem()

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

        while time.time() - start_time < time_limit:
            leaf_node = self._select(root_node)
            self._expand(leaf_node)
            simulation_result = self._simulate(leaf_node)
            self._backpropagate(leaf_node, simulation_result)

        # Choose the action leading to the most visited child node
        best_action = max(root_node.children, key=lambda child: child.visits).action
        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a node to expand using UCT.

        Args:
            node (MCTSNode): The current node.

        Returns:
            MCTSNode: The selected node.
        """
        while node.children:
            best_node = max(
                node.children,
                key=lambda child: (child.value / child.visits if child.visits > 0 else float('inf')) +
                                  self.mcts_c * np.sqrt(np.log(node.visits) / (child.visits + 1))
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
        for action in actions:
            next_state = self.search_problem.transition(node.state, action)
            child_node = MCTSNode(state=next_state, parent=node, action=action)
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

# Helper functions to load models
def load_value_model(path: str, input_size: int) -> nn.Module:
    model = ValueNetwork(input_size)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_policy_model(path: str, input_size: int, board_size: int) -> nn.Module:
    model = PolicyNetwork(input_size, board_size)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Example instantiation
if __name__ == "__main__":
    board_size = 5
    feature_size = board_size ** 2 * 4  # Example feature size based on board size

    value_model_path = "value_model.pt"
    policy_model_path = "policy_model.pt"

    value_model = load_value_model(value_model_path, feature_size)
    policy_model = load_policy_model(policy_model_path, feature_size, board_size)

    final_agent = FinalAgent(value_model, policy_model, board_size=board_size)

    # Example game state
    game_state = GoState(board_size)
    time_limit = 1.0  # 1 second per move

    action = final_agent.get_move(game_state, time_limit)
    print(f"FinalAgent chose action: {action}")

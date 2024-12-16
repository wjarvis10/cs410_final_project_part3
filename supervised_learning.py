import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from go_search_problem import GoProblem, GoState
from heuristic_go_problems import GoProblemLearnedHeuristic, GoProblemSimpleHeuristic
from agents import GreedyAgent, RandomAgent, MCTSAgent, GameAgent, MinimaxAgent, AlphaBetaAgent, IterativeDeepeningAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
from game_runner import run_many
import pickle

torch.set_default_tensor_type(torch.FloatTensor)
################################################### FUNCTIONS ##################################################

def load_dataset(path: str):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save_model(path: str, model):
    """
    Save model to a file
    Input:
        path: path to save model to
        model: Pytorch model to save
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

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
    # TODO: Encode game_state into a list of features
    features = np.array(game_state.get_board()).flatten()
    return features

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

def train_value_network(dataset, num_epochs, learning_rate):
    """
    Train a value network on the provided dataset.

    Input:
        dataset: list of (state, action, result) tuples
        num_epochs: number of epochs to train for
        learning_rate: learning rate for gradient descent
    Output:
        model: trained model
    """
    # Make sure dataset is shuffled for better performance
    random.shuffle(dataset)
    
    # You may find it useful to create train/test sets to better track performance/overfit/underfit
    # train_size = int(0.8 * len(dataset))
    # train_data = dataset[:train_size]
    # test_data = dataset[train_size:]

    # TODO: Create model
    input_size = len(get_features(dataset[0][0]))
    model = ValueNetwork(input_size)

    # TODO: Specify Loss Function - use MSE bc continuous outcome value
    loss_function = nn.MSELoss()

    # You can use Adam, which is stochastic gradient descent with ADAptive Momentum
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 32

    for epoch in range(num_epochs):

        batch_loss = 0.0
        batch_counter = 0

        # state = (state, action, outcome)
        for data_point in dataset:
            state = data_point[0]
            features = get_features(state)
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # TODO: What should the desired output of the value network be?
            # Note: You will have to convert the label to a torch tensor to use with torch's loss functions
            outcome = data_point[2]
            label = torch.tensor(outcome)

            # TODO: Get model prediction of value
            prediction = model(features_tensor)

            # TODO: Compute Loss for data point
            loss = loss_function(prediction, label)
            batch_loss += loss
            batch_counter += 1

            if batch_counter % batch_size == 0:
                # Call backward to run backward pass and compute gradients
                batch_loss.backward()

                # Run gradient descent step with optimizer
                optimizer.step()

                # Reset gradient for next batch
                optimizer.zero_grad()
                batch_loss = 0.0
        print("Epoch: ", epoch)
        print("Loss: ", batch_loss)
    return model

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

def train_policy_network(dataset, num_epochs, learning_rate):
    """
    Train a policy network on the provided dataset.

    Input:
        dataset: list of (state, action, result) tuples
        num_epochs: number of epochs to train for
        learning_rate: learning rate for gradient descent
    Output:
        model: trained model
    """
    random.shuffle(dataset)

    # TODO: Create model
    feature_size =  len(get_features(dataset[0][0]))
    model = PolicyNetwork(feature_size)

    # TODO: Specify Loss Function
    loss_function = nn.CrossEntropyLoss()

    # You can use Adam, which is stochastic gradient descent with ADAptive Momentum
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        # data = (state, action, outcome)
        for data_point in dataset:
            optimizer.zero_grad()

            # TODO: Get features from state and convert features to torch tensor
            state = data_point[0]
            action = data_point[1]

            features = get_features(state)
            features_tensor = torch.tensor(features, dtype=torch.float32)

            # TODO: What should the desired output of the value network be?
            # Note: You will have to convert the label to a torch tensor to use with torch's loss functions
            label = torch.tensor(action)

            # TODO: Get model estimate of value
            prediction = model(features_tensor)

            # TODO: Compute Loss for data point
            loss = loss_function(prediction, label)

            loss.backward()

            optimizer.step()

        print("Epoch: ", epoch)
        print("Loss: ", loss)
        
    return model

################################################## TRAIN MODELS #########################################################


dataset_5x5 = load_dataset('dataset_5x5.pkl')
# dataset_9x9 = load_dataset('9x9_dataset.pkl')

value_model = train_value_network(dataset_5x5, 10, 1e-4)
save_model("value_model_2.pt", value_model)

policy_net = train_policy_network(dataset_5x5, 10, 1e-4)
save_model("policy_model_2.pt", policy_net)
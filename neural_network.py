import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for the RL agent.

    This is a feed-forward neural network that takes the game state as input
    and outputs the expected Q-value for each possible action. The architecture
    consists of fully connected layers with ReLU activation functions for the
    hidden layers.
    """

    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the neural network layers.

        Args:
            state_size (int): The dimension of the input state space. This corresponds
                              to the number of features in the state vector.
            action_size (int): The number of possible actions the agent can take,
                               which is the output dimension of the network.
        """
        super(DQN, self).__init__()

        # Define the network layers. A simple multi-layer perceptron (MLP).
        # The number of neurons in hidden layers (256, 128) are chosen to provide
        # sufficient capacity for learning the game's dynamics without being
        # excessively large for this problem.

        # Input layer: state_size -> 256 neurons
        self.fc1 = nn.Linear(state_size, 256)
        
        # Hidden layer: 256 -> 128 neurons
        self.fc2 = nn.Linear(256, 128)
        
        # Output layer: 128 -> action_size neurons
        # The output neurons correspond to the Q-values for each action.
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        This method takes a state tensor and passes it through the network layers
        to produce Q-values.

        Args:
            state (torch.Tensor): The input tensor representing one or more game states.
                                  Shape should be (batch_size, state_size).

        Returns:
            torch.Tensor: The output tensor containing the Q-values for each action
                          for each state in the batch. Shape: (batch_size, action_size).
        """
        # Pass the input state through the first fully connected layer,
        # followed by a ReLU activation function.
        x = F.relu(self.fc1(state))
        
        # Pass the result through the second hidden layer, also with ReLU activation.
        x = F.relu(self.fc2(x))
        
        # The final layer is the output layer. It has no activation function,
        # as it represents the raw, unbounded Q-values for each action.
        q_values = self.fc3(x)
        
        return q_values

class ActorCriticNet(nn.Module):
    """Simple actor-critic network producing policy logits and state value."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


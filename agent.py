import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os

# Internal project imports
import config
from neural_network import DQN

# --- Device Setup ---
# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a named tuple to represent a single transition in our environment for clarity
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class Agent:
    """
    Implements a Deep Q-Network (DQN) agent for playing the Pong game.

    This class encapsulates the neural networks (policy and target), the experience
    replay memory, the action selection policy (epsilon-greedy), and the learning
    algorithm.
    """

    def __init__(self, agent_id: str):
        """
        Initializes the DQN Agent.

        Args:
            agent_id (str): A unique identifier for the agent (e.g., 'agent1' or 'agent2').
                            Used for saving and loading models.
        """
        self.agent_id = agent_id
        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE

        # --- Q-Network Initialization ---
        # Policy Network: The network we are actively training
        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        # Target Network: A clone of the policy network for stabilizing learning
        self.target_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # --- Optimizer and Loss Function ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        # Smooth L1 Loss is often more stable than MSE for Q-learning
        self.loss_fn = nn.SmoothL1Loss()

        # --- Replay Memory ---
        # A deque (double-ended queue) for storing experiences efficiently
        self.memory = deque(maxlen=config.MEMORY_SIZE)

        # --- Epsilon-Greedy Policy ---
        # Epsilon is the probability of choosing a random action (exploration)
        self.epsilon = config.EPSILON_START

    def choose_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value predicted by the policy
        network is chosen (exploitation).

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The chosen action (0: STAY, 1: UP, 2: DOWN, 3: SWING).
        """
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: choose the best action based on the policy network
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.policy_net.eval()  # Set network to evaluation mode for inference
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train() # Set it back to train mode for subsequent learning
            # Get the action with the highest Q-value
            return torch.argmax(action_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state (np.ndarray): The starting state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The resulting state.
            done (bool): A flag indicating if the episode has ended.
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def learn(self):
        """
        Trains the policy network using a batch of experiences from the replay memory.

        This method samples a minibatch, calculates the target Q-values using the
        target network, computes the loss against the policy network's predictions,
        and performs a gradient descent step.
        """
        if len(self.memory) < config.BATCH_SIZE:
            return  # Not enough experiences in memory to train

        # Sample a random minibatch of experiences from memory
        experiences = random.sample(self.memory, config.BATCH_SIZE)
        batch = Experience(*zip(*experiences))

        # Convert batch of experiences to PyTorch tensors
        states = torch.from_numpy(np.vstack(batch.state)).float().to(device)
        actions = torch.from_numpy(np.vstack(batch.action)).long().to(device)
        rewards = torch.from_numpy(np.vstack(batch.reward)).float().to(device)
        next_states = torch.from_numpy(np.vstack(batch.next_state)).float().to(device)
        dones = torch.from_numpy(np.vstack(batch.done).astype(np.uint8)).float().to(device)

        # 1. Get Q-values for current states from the policy network
        # We need to select the Q-value for the action that was actually taken.
        # policy_net(states) -> (batch_size, action_size)
        # .gather(1, actions) -> selects the specific action's Q-value for each state
        current_q_values = self.policy_net(states).gather(1, actions)

        # 2. Calculate target Q-values for the next states from the target network
        with torch.no_grad(): # We don't need gradients for the target network
            # .max(1)[0] gives the max Q-value for each next_state
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # The future reward is zero if the episode is done (terminal state)
            target_q_values = rewards + (config.GAMMA * next_q_values * (1 - dones))

        # 3. Compute the loss between current and target Q-values
        loss = self.loss_fn(current_q_values, target_q_values)

        # 4. Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent them from exploding, a common practice in RL
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        """
        Updates the target network's weights by copying them from the policy network.
        This is a "hard" update, performed periodically.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Decays the epsilon value according to the configured decay rate.
        This reduces exploration over time as the agent becomes more confident.
        """
        self.epsilon = max(config.EPSILON_END, config.EPSILON_DECAY * self.epsilon)

    def save_model(self, episode: int):
        """
        Saves the policy network's weights to a file.

        Args:
            episode (int): The current episode number, used in the filename.
        """
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        filename = f"{self.agent_id}_episode_{episode}.pth"
        path = os.path.join(config.MODEL_PATH, filename)
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model for {self.agent_id} saved to {path}")

    def load_model(self, path: str):
        """
        Loads the policy network's weights from a file into both networks.

        Args:
            path (str): The file path to the saved model weights.
        """
        if os.path.exists(path):
            try:
                self.policy_net.load_state_dict(torch.load(path, map_location=device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.policy_net.train()
                self.target_net.eval()
                print(f"Model for {self.agent_id} loaded from {path}")
            except Exception as e:
                print(f"Error loading model from {path}. Starting fresh. Error: {e}")
        else:
            print(f"No model found at {path}, starting from scratch.")
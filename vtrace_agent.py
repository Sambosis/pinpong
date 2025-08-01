import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from neural_network import ActorCriticNet

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VTraceExperience = namedtuple(
    'VTraceExperience',
    ('state', 'action', 'reward', 'next_state', 'done', 'log_prob')
)

class VTraceAgent:
    """Trust-region actor-critic agent using V-trace off-policy corrections."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE

        self.network = ActorCriticNet(self.state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)

        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.epsilon = config.EPSILON_START
        self.last_log_prob = 0.0

    def choose_action(self, state: np.ndarray) -> int:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits, _ = self.network(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        dist = torch.distributions.Categorical(probs)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            log_prob = float(torch.log(torch.tensor(1.0 / self.action_size)))
        else:
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=device)).item()
        self.last_log_prob = log_prob
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(VTraceExperience(state, action, reward, next_state, done, self.last_log_prob))

    def learn(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        experiences = random.sample(self.memory, config.BATCH_SIZE)
        batch = VTraceExperience(*zip(*experiences))

        states = torch.from_numpy(np.vstack(batch.state)).float().to(device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        next_states = torch.from_numpy(np.vstack(batch.next_state)).float().to(device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
        behavior_log_probs = torch.tensor(batch.log_prob, dtype=torch.float32, device=device)

        logits, values = self.network(states)
        dist = torch.distributions.Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = values.squeeze()

        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.squeeze()

        rho = torch.exp(action_log_probs - behavior_log_probs)
        rho_bar = torch.clamp(rho, max=config.VTRACE_RHO_CLIP)
        c = torch.clamp(rho, max=config.VTRACE_C_CLIP)

        # Calculate 1-step temporal difference error
        td_error = rewards + config.GAMMA * next_values * (1 - dones) - values

        # V-trace value target uses rho_bar clipped importance ratios
        deltas = rho_bar * td_error
        value_targets = values + deltas

        value_loss = (value_targets.detach() - values).pow(2).mean()

        # V-trace policy gradient advantage uses c clipped importance ratios
        pg_advantage = (c * td_error).detach()
        policy_loss = -(action_log_probs * pg_advantage).mean()
        policy_loss -= config.ENTROPY_BETA * entropy

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 40.0)
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END, config.EPSILON_DECAY * self.epsilon)

    def update_target_network(self):
        # Actor-critic does not use a target network; method kept for API parity.
        pass

    def save_model(self, episode: int):
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        filename = f"{self.agent_id}_episode_{episode}.pth"
        path = os.path.join(config.MODEL_PATH, filename)
        torch.save(self.network.state_dict(), path)
        print(f"Model for {self.agent_id} saved to {path}")

    def load_model(self, path: str):
        if os.path.exists(path):
            try:
                self.network.load_state_dict(torch.load(path, map_location=device))
                print(f"Model for {self.agent_id} loaded from {path}")
            except Exception as e:
                print(f"Error loading model from {path}. Starting fresh. Error: {e}")
        else:
            print(f"No model found at {path}, starting from scratch.")

"""
config.py

This file contains all game constants, hyperparameters, and configuration settings
for the Deep Q-Learning Pong game. Centralizing these parameters makes them
easy to adjust and tune.
"""

# ---------------------------- #
# Game & Display Settings
# ---------------------------- #
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 608
WIN_WIDTH = SCREEN_WIDTH  # Alias for compatibilityWIN_HEIGHT = SCREEN_HEIGHT  # Alias for compatibility
GAME_CAPTION = "RL Flipper Pong"
FPS =90  # Frames per second for rendering
WINNING_SCORE = 6  # Score needed to win a game

# ---------------------------- #
# Colors
# ---------------------------- #
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PADDLE_COLOR = BLUE
BALL_COLOR = RED
# ---------------------------- #
# Physics Constants
# ---------------------------- #
# Defines the fixed time interval used for each physics update (e.g., movement calculations).
# By synchronizing this with FPS (e.g., 1.0 / 60.0 for 60 FPS), we ensure that the game's
# physics are consistent and deterministic, which is crucial for stable RL training.
# This decouples game speed from rendering speed; the game will run at the same speed
# regardless of whether the visual frame rate is high or low. A higher physics update
# rate (smaller timestep) prevents "tunneling" where fast-moving objects can pass
# through thin obstacles.
FIXED_TIMESTEP = 1.0 / 45.0  # Timestep for physics updates, independent of FPS

# ---------------------------- #
# Paddle Constants
# ---------------------------- #
PADDLE_WIDTH = 22
PADDLE_HEIGHT = 100
PADDLE_SPEED = 11.0
PADDLE_WALL_OFFSET = 24  # Distance from the side walls

# Flipper/Swing Mechanics
PADDLE_MAX_SWING_ANGLE = 42  # Max rotation in degrees
PADDLE_SWING_DURATION = 0.1  # Duration of the swing animation in seconds

# ---------------------------- #
# Ball Constants
# ---------------------------- #
BALL_RADIUS = 10
BALL_INITIAL_SPEED_X = 300.0
BALL_INITIAL_SPEED_Y = 200.0
BALL_MAX_SPEED = 900.0  # Maximum speed magnitude for the ball
BALL_SPEED_INCREASE_FACTOR = 1.51  # Factor by which speed increases on paddle hit
BALL_SPEED_INITIAL = 550.0  # Initial ball speed (fast enough to reach paddles)
BALL_SPEED_SWING = 900.0   # Ball speed after swing hit
MAX_BOUNCE_ANGLE = 42     # Maximum bounce angle in degrees

# Anti-stall parameters (to prevent vertical bouncing)
BALL_MIN_X_VELOCITY = 80.0  # Minimum acceptable x velocity magnitude
BALL_VERTICAL_RATIO_THRESHOLD = 5.0  # If |vy| / |vx| > this, ball is too vertical
BALL_X_BOOST_PER_FRAME = 2.0  # Small constant boost applied per frame when ball is too vertical

# ---------------------------- #
# Reinforcement Learning Hyperparameters
# ---------------------------- #
# State & Action Space
# State: [ball_x, ball_y, ball_vx, ball_vy, player1_y, player2_y, player1_swing_timer, player2_swing_timer]
STATE_SIZE = 8
# Actions: [STAY, UP, DOWN, SWING]
ACTION_SIZE = 4

# DQN Agent Settings
MEMORY_SIZE = 100000        # Replay buffer size
BATCH_SIZE = 512           # Minibatch size for training
GAMMA = 0.99               # Discount factor for future rewards
LEARNING_RATE = 0.00005     # Learning rate for the Adam optimizer

# Epsilon-Greedy Policy for Exploration
EPSILON_START = 0.49      # Initial exploration rate
EPSILON_END = 0.01         # Minimum exploration rate
EPSILON_DECAY = 0.99        # Decay rate for epsilon per episode

# Target Network Update
TARGET_UPDATE_FREQUENCY = 10 # Update target network every N episodes

# ---------------------------- #
# Reward Structure
# ---------------------------- #
REWARD_SCORE = 20.0          # Reward for scoring a point
REWARD_MISS = -4.0        # Reward for conceding a point
REWARD_HIT = 3.0          # Small positive reward for hitting the ball
REWARD_SWING = -0.9      # Small penalty for swinging paddle (discourages unnecessary swings)
REWARD_GAME_WIN = 40.0     # Large reward for winning the entire game (reaching WINNING_SCORE)
REWARD_GAME_LOSE = -5.0   # Large penalty for losing the entire game
REWARD_WIN = REWARD_SCORE  # Alias for winning reward
REWARD_LOSE = REWARD_MISS  # Alias for losing penalty
# ---------------------------- #
# Training & Model Settings
# ---------------------------- #
NUM_EPISODES = 12000       # Total number of games to play for training
MAX_STEPS_PER_EPISODE = 1200  # Maximum steps per episode (60 seconds at 30 FPS)
DISPLAY_EVERY = 5         # Render every Nth game
SAVE_MODEL_EVERY = 100    # Save trained models every N episodes
MODEL_PATH = "models/"    # Directory to save/load models
LOAD_MODEL = False        # Set to True to load a pre-trained model
MODEL_TO_LOAD_P1 = "models/agent1_episode_3600.pth" # Example path
MODEL_TO_LOAD_P2 = "models/agent2_episode_3600.pth" # Example path
# V-trace / Actor-Critic Settings

# ENTROPY_BETA: Coefficient for the entropy bonus in the policy loss.
# The entropy bonus encourages exploration by penalizing the policy for being too
# certain. A higher value promotes more random actions, which can help the agent
# discover better strategies, especially early in training.
# - Increasing this value leads to more exploration but can prevent the policy
#   from converging to an optimal solution if too high.
# - Decreasing this value leads to more exploitation, faster convergence, but
#   risks getting stuck in a local optimum.
ENTROPY_BETA = 0.02

# VTRACE_RHO_CLIP: The clipping threshold for the rho importance sampling ratio,
# used for the V-trace value target calculation. This corrects for the difference
# between the target policy and the behavior policy (off-policy correction).
# Clipping prevents the value updates from becoming too large and unstable.
# - Increasing this value allows for more aggressive off-policy corrections,
#   which can speed up learning but may increase variance and instability.
# - Decreasing this value makes the value updates more conservative and stable,
#   but can slow down learning. A value of 1.0 is a common default.
VTRACE_RHO_CLIP = 1.0

# VTRACE_C_CLIP: The clipping threshold for the c importance sampling ratio,
# used for the V-trace policy gradient calculation. This also corrects for the
# off-policy data. Clipping stabilizes the policy gradient updates.
# - Increasing this value allows the policy to change more drastically based on
#   past experiences, which can be faster but riskier.
# - Decreasing this value makes policy updates smaller and more stable, but
#   potentially slower. A value of 1.0 is a common default.
VTRACE_C_CLIP = 1.0

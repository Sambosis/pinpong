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
SCREEN_HEIGHT = 600
WIN_WIDTH = SCREEN_WIDTH  # Alias for compatibilityWIN_HEIGHT = SCREEN_HEIGHT  # Alias for compatibility
GAME_CAPTION = "RL Flipper Pong"
FPS =90  # Frames per second for rendering
WINNING_SCORE = 3  # Score needed to win a game

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
FIXED_TIMESTEP = 1.0 / 50.0  # Timestep for physics updates, independent of FPS

# ---------------------------- #
# Paddle Constants
# ---------------------------- #
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 100
PADDLE_SPEED = 14.0
PADDLE_WALL_OFFSET = 20  # Distance from the side walls

# Flipper/Swing Mechanics
PADDLE_MAX_SWING_ANGLE = 38  # Max rotation in degrees
PADDLE_SWING_DURATION = 0.2  # Duration of the swing animation in seconds

# ---------------------------- #
# Ball Constants
# ---------------------------- #
BALL_RADIUS = 10
BALL_INITIAL_SPEED_X = 300.0
BALL_INITIAL_SPEED_Y = 200.0
BALL_MAX_SPEED = 900.0  # Maximum speed magnitude for the ball
BALL_SPEED_INCREASE_FACTOR = 1.51  # Factor by which speed increases on paddle hit
BALL_SPEED_INITIAL = 450.0  # Initial ball speed (fast enough to reach paddles)
BALL_SPEED_SWING = 700.0   # Ball speed after swing hit
MAX_BOUNCE_ANGLE = 42     # Maximum bounce angle in degrees

# Anti-stall parameters (to prevent vertical bouncing)
BALL_MIN_X_VELOCITY = 50.0  # Minimum acceptable x velocity magnitude
BALL_VERTICAL_RATIO_THRESHOLD = 8.0  # If |vy| / |vx| > this, ball is too vertical
BALL_X_BOOST_PER_FRAME = 1.0  # Small constant boost applied per frame when ball is too vertical

# ---------------------------- #
# Reinforcement Learning Hyperparameters
# ---------------------------- #
# State & Action Space
# State: [ball_x, ball_y, ball_vx, ball_vy, player1_y, player2_y, player1_swing_timer, player2_swing_timer]
STATE_SIZE = 8
# Actions: [STAY, UP, DOWN, SWING]
ACTION_SIZE = 4

# DQN Agent Settings
MEMORY_SIZE = 50000        # Replay buffer size
BATCH_SIZE = 128           # Minibatch size for training
GAMMA = 0.98               # Discount factor for future rewards
LEARNING_RATE = 0.0001     # Learning rate for the Adam optimizer

# Epsilon-Greedy Policy for Exploration
EPSILON_START = 0.8      # Initial exploration rate
EPSILON_END = 0.01         # Minimum exploration rate
EPSILON_DECAY = 0.993     # Decay rate for epsilon per episode

# Target Network Update
TARGET_UPDATE_FREQUENCY = 10 # Update target network every N episodes

# ---------------------------- #
# Reward Structure
# ---------------------------- #
REWARD_WIN = 4.0          # Reward for scoring a point
REWARD_LOSE = -4.0        # Reward for conceding a point
REWARD_HIT = 8.0          # Small positive reward for hitting the ball
REWARD_SWING = -0.02      # Small penalty for swinging paddle (discourages unnecessary swings)
REWARD_GAME_WIN = 6.0     # Large reward for winning the entire game (reaching WINNING_SCORE)
REWARD_GAME_LOSE = -6.0   # Large penalty for losing the entire game

# ---------------------------- #
# Training & Model Settings
# ---------------------------- #
NUM_EPISODES = 20000       # Total number of games to play for training
MAX_STEPS_PER_EPISODE = 4200  # Maximum steps per episode (60 seconds at 30 FPS)
DISPLAY_EVERY = 2         # Render every Nth game
SAVE_MODEL_EVERY = 100    # Save trained models every N episodes
MODEL_PATH = "models/"    # Directory to save/load models
LOAD_MODEL = True        # Set to True to load a pre-trained model
MODEL_TO_LOAD_P1 = "models/agent1_episode_300.pth" # Example path
MODEL_TO_LOAD_P2 = "models/agent2_episode_300.pth" # Example path
# V-trace / Actor-Critic Settings
ENTROPY_BETA = 0.01
VTRACE_RHO_CLIP = 1.0
VTRACE_C_CLIP = 1.0

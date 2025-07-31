import pygame
import numpy as np
import math

# Internal project imports
import config
from ball import Ball
from paddle import Paddle

class PongGame:
    """
    Manages the overall game state, physics, rendering, and agent interactions.

    This class orchestrates the game loop, including handling player actions,
    updating the physics of the ball and paddles with a fixed timestep,
    detecting collisions, managing the scoring system, and providing a
    normalized state representation for the reinforcement learning agents.
    """

    def __init__(self, render_mode: bool = False):
        """
        Initializes the Pong game environment.

        Args:
            render_mode (bool): If True, initializes Pygame for graphical display.
                                Set to False for faster, non-displayed training.
        """
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode:
            pygame.init()
            pygame.display.set_caption(config.GAME_CAPTION)
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Create game objects
        self.paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
        self.paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
        self.ball = Ball()

        # Game state variables
        self.score1 = 0
        self.score2 = 0
        self.time_accumulator = 0.0

    def reset(self) -> np.ndarray:
        """
        Resets the game to its initial state for a new episode.

        This involves resetting scores, and the positions and states of the
        paddles and ball.

        Returns:
            np.ndarray: The initial state of the game environment.
        """
        self.score1 = 0
        self.score2 = 0
        self.paddle1.reset()
        self.paddle2.reset()
        # Reset ball, ensuring it serves towards a random side initially
        self.ball.reset(scored_left=np.random.choice([True, False]))
        self.time_accumulator = 0.0
        return self._get_state()

    def step(self, action1: int, action2: int, dt: float) -> tuple[np.ndarray, float, float, bool]:
        """
        Executes one time step of the game.

        Args:
            action1 (int): The action chosen by agent 1.
            action2 (int): The action chosen by agent 2.
            dt (float): The delta time since the last frame, in seconds.

        Returns:
            tuple: A tuple containing:
                - next_state (np.ndarray): The state of the game after the step.
                - reward1 (float): The reward for agent 1.
                - reward2 (float): The reward for agent 2.
                - done (bool): True if the game has ended, False otherwise.
        """
        self._apply_actions(action1, action2)
        
        # Update physics using a fixed timestep for consistency
        self.time_accumulator += dt
        while self.time_accumulator >= config.FIXED_TIMESTEP:
            self._update_physics(config.FIXED_TIMESTEP)
            self.time_accumulator -= config.FIXED_TIMESTEP

        # Check for scoring and collisions, and calculate rewards
        hit_reward1, hit_reward2 = self._handle_collisions()
        score_reward1, score_reward2 = self._handle_scoring()
        
        reward1 = hit_reward1 + score_reward1
        reward2 = hit_reward2 + score_reward2

        # Check if the game is over
        done = (self.score1 >= config.WINNING_SCORE or
                self.score2 >= config.WINNING_SCORE)

        next_state = self._get_state()

        return next_state, reward1, reward2, done

    def _apply_actions(self, action1: int, action2: int):
        """Maps integer actions to paddle movements and swings."""
        # Action mapping: 0: STAY, 1: UP, 2: DOWN, 3: SWING
        # Player 1 (Left Paddle)
        if action1 == 1:
            self.paddle1.move(-1)  # -1 for UP
        elif action1 == 2:
            self.paddle1.move(1)   # 1 for DOWN
        elif action1 == 3:
            self.paddle1.swing()

        # Player 2 (Right Paddle)
        if action2 == 1:
            self.paddle2.move(-1)
        elif action2 == 2:
            self.paddle2.move(1)
        elif action2 == 3:
            self.paddle2.swing()

    def _update_physics(self, fixed_dt: float):
        """Updates all game objects by one fixed time step."""
        self.paddle1.update(fixed_dt)
        self.paddle2.update(fixed_dt)
        self.ball.move(fixed_dt)
        self.ball.handle_wall_collision()
        # Increase ball speed each step until a point is scored
        self.ball.increase_speed()

    def _handle_collisions(self) -> tuple[float, float]:
        """
        Checks for and handles ball-paddle collisions.

        Returns:
            tuple[float, float]: Rewards for agent 1 and agent 2 for hitting the ball.
        """
        reward1, reward2 = 0.0, 0.0
        
        # AABB broad-phase collision check
        if self.ball.rect.colliderect(self.paddle1.rect):
            hit_occured = self.ball.handle_paddle_collision(self.paddle1)
            if hit_occured:
                reward1 = config.REWARD_HIT
        
        if self.ball.rect.colliderect(self.paddle2.rect):
            hit_occured = self.ball.handle_paddle_collision(self.paddle2)
            if hit_occured:
                reward2 = config.REWARD_HIT
                
        return reward1, reward2

    def _handle_scoring(self) -> tuple[float, float]:
        """
        Checks if a player has scored, updates scores, and resets the ball.

        Returns:
            tuple[float, float]: Rewards for agent 1 and agent 2 from scoring.
        """
        reward1, reward2 = 0.0, 0.0
        
        scorer = self.ball.check_score()
        if scorer:
            if scorer == 'left':  # Left player (Agent 1) scored
                self.score1 += 1
                reward1 = config.REWARD_WIN
                reward2 = config.REWARD_LOSE
                self.ball.reset(scored_left=False)  # Serve to the loser (right)
            elif scorer == 'right':  # Right player (Agent 2) scored
                self.score2 += 1
                reward1 = config.REWARD_LOSE
                reward2 = config.REWARD_WIN
                self.ball.reset(scored_left=True)   # Serve to the loser (left)

        return reward1, reward2

    def _get_state(self) -> np.ndarray:
        """
        Generates the normalized state vector for the RL agents.

        The state is absolute (world-centric), not relative to each player.
        This provides a complete picture of the environment. Normalization
        is crucial for stable neural network training.

        State Vector Composition:
        [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y, paddle1_swing_timer, paddle2_swing_timer]

        Returns:
            np.ndarray: A numpy array representing the normalized game state.
        """
        # Normalize ball position (0 to 1)
        ball_x_norm = self.ball.rect.centerx / config.SCREEN_WIDTH
        ball_y_norm = self.ball.rect.centery / config.SCREEN_HEIGHT

        # Normalize ball velocity (-1 to 1)
        ball_vx_norm = np.clip(self.ball.vx / config.BALL_MAX_SPEED, -1, 1)
        ball_vy_norm = np.clip(self.ball.vy / config.BALL_MAX_SPEED, -1, 1)

        # Normalize paddle positions (0 to 1)
        p1_y_norm = self.paddle1.rect.centery / config.SCREEN_HEIGHT
        p2_y_norm = self.paddle2.rect.centery / config.SCREEN_HEIGHT

        # Normalize swing timers (0 to 1, indicates swing progress)
        p1_swing_norm = self.paddle1.swing_timer / config.PADDLE_SWING_DURATION
        p2_swing_norm = self.paddle2.swing_timer / config.PADDLE_SWING_DURATION

        state = np.array([
            ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm,
            p1_y_norm, p2_y_norm, p1_swing_norm, p2_swing_norm
        ], dtype=np.float32)

        return state

    def render(self, game_num: int):
        """
        Draws the current game state to the screen.

        Args:
            game_num (int): The current episode number to display.
        """
        if not self.render_mode:
            return

        self.screen.fill(config.BLACK)

        # Draw paddles and ball
        self.paddle1.draw(self.screen)
        self.paddle2.draw(self.screen)
        self.ball.draw(self.screen)

        # Draw a center line
        pygame.draw.aaline(self.screen, config.GRAY,
                           (config.SCREEN_WIDTH / 2, 0),
                           (config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT))

        # Render scores
        score_text = self.font.render(f"{self.score1}  -  {self.score2}", True, config.WHITE)
        self.screen.blit(score_text, (config.SCREEN_WIDTH / 2 - score_text.get_width() / 2, 10))
        
        # Render game number
        game_text = self.font.render(f"Game: {game_num}", True, config.GRAY)
        self.screen.blit(game_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def close(self):
        """Cleans up and closes the Pygame window."""
        if self.render_mode:
            pygame.quit()
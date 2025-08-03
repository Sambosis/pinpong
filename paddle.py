import pygame
import math
import numpy as np

# Internal project import
import config

class Paddle:
    """
    Represents a player's paddle with vertical movement and pinball-like swing mechanics.

    The paddle can move up and down along the edge of the screen and perform a
    'swing' action that rotates it toward the center of the screen for a brief
    period. This rotation affects the ball's bounce angle and speed.
    """

    def __init__(self, x: float, side: str):
        """
        Initializes the Paddle object.

        Args:
            x (float): The initial x-coordinate for the paddle's top-left corner.
            side (str): The side of the screen the paddle is on. Must be 'left' or 'right'.
                        This determines the direction of the swing.
        """
        if side not in ['left', 'right']:
            raise ValueError("Side must be 'left' or 'right'")

        self.initial_x = x
        self.initial_y = config.SCREEN_HEIGHT / 2 - config.PADDLE_HEIGHT / 2
        self.side = side

        self.width = config.PADDLE_WIDTH
        self.height = config.PADDLE_HEIGHT
        
        # The un-rotated rectangular area of the paddle
        self.rect = pygame.Rect(self.initial_x, self.initial_y, self.width, self.height)
        
        self.color = config.PADDLE_COLOR
        self.speed = config.PADDLE_SPEED

        # Swing related attributes
        self.angle = 0.0  # Current rotation angle in degrees
        self.angular_velocity = 0.0  # Degrees per second, useful for physics
        self.is_swinging = False
        self.swing_duration = config.PADDLE_SWING_DURATION
        self.max_swing_angle = config.PADDLE_MAX_SWING_ANGLE
        self.swing_timer = 0.0

    def reset(self):
        """Resets the paddle to its initial position and state."""
        self.rect.y = self.initial_y
        self.is_swinging = False
        self.swing_timer = 0.0
        self.angle = 0.0
        self.angular_velocity = 0.0

    def move(self, direction: int):
        """
        Moves the paddle vertically based on a discrete action.

        Args:
            direction (int): The direction to move. 1 for down, -1 for up, 0 for stay.
        """
        if direction not in [-1, 0, 1]:
            return  # Ignore invalid directions
        
        self.rect.y += direction * self.speed
        
        # Clamp the paddle's position to stay within the screen boundaries
        self.rect.y = max(0, min(self.rect.y, config.SCREEN_HEIGHT - self.height))

    def swing(self):
        """Initiates the swing action if the paddle is not already swinging."""
        if not self.is_swinging:
            self.is_swinging = True
            self.swing_timer = self.swing_duration

    def update(self, dt: float):
        """
        Updates the paddle's state, primarily for handling the swing animation.

        This should be called every frame or physics step.

        Args:
            dt (float): The time delta since the last update, in seconds.
        """
        if dt == 0:
            return  # Avoid division by zero if time has not passed

        old_angle = self.angle

        if self.is_swinging:
            self.swing_timer -= dt
            if self.swing_timer <= 0:
                # End of swing animation
                self.is_swinging = False
                self.swing_timer = 0.0
                self.angle = 0.0
            else:
                # Calculate the current angle of the swing using a sine wave
                # for a smooth out-and-back motion.
                progress = 1.0 - (self.swing_timer / self.swing_duration)
                swing_factor = math.sin(progress * math.pi)
                
                self.angle = self.max_swing_angle * swing_factor
                
                # Right paddle swings in the opposite direction (negative angle)
                if self.side == 'right':
                    self.angle *= -1
        else:
            # Ensure angle is zero when not swinging
            self.angle = 0.0
        
        # Calculate angular velocity (degrees per second) for physics calculations
        self.angular_velocity = (self.angle - old_angle) / dt

    def get_corners(self) -> list[np.ndarray]:
        """
        Calculates the world coordinates of the four corners of the rotated paddle.
        This is essential for accurate collision detection and rendering.

        Returns:
            list[np.ndarray]: A list of four 2D numpy arrays, each representing a corner's (x, y) coords.
        """
        center = np.array(self.rect.center)
        half_w = self.width / 2
        half_h = self.height / 2

        # Corner points relative to the paddle's center (0,0) before rotation
        local_corners = [
            np.array([-half_w, -half_h]),  # Top-left
            np.array([ half_w, -half_h]),  # Top-right
            np.array([ half_w,  half_h]),  # Bottom-right
            np.array([-half_w,  half_h]),  # Bottom-left
        ]

        # Create 2D rotation matrix
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a],
                               [sin_a,  cos_a]])

        # Rotate each corner and translate it to its world position by adding the center coordinates
        world_corners = [center + (rot_matrix @ p) for p in local_corners]
        
        return world_corners

    def draw(self, screen: pygame.Surface):
        """
        Draws the paddle on the given screen as a rotated polygon.

        Args:
            screen (pygame.Surface): The Pygame surface to draw on.
        """
        # Get the rotated corners and draw a polygon
        corners = self.get_corners()
        # pygame.draw.polygon expects a list of tuples, so we convert the numpy arrays
        pygame.draw.polygon(screen, self.color, [tuple(p) for p in corners])

    def get_state(self) -> np.ndarray:
        """
        Gets the normalized state of the paddle for the reinforcement learning agent.

        Returns:
            np.ndarray: A numpy array containing the paddle's normalized state
                        [normalized_y_position, is_swinging_flag].
        """
        # Normalize y-position to be between -1 (top) and 1 (bottom) relative to screen center
        norm_y = (self.rect.centery - config.SCREEN_HEIGHT / 2) / (config.SCREEN_HEIGHT / 2)
        
        # Swing state as a binary flag (0.0 for not swinging, 1.0 for swinging)
        swing_state = 1.0 if self.is_swinging else 0.0

        return np.array([norm_y, swing_state], dtype=np.float32)
    
    def get_surface_angle(self):
        """
        Returns the current surface angle of the paddle for collision calculations.
        
        Returns:
            float: The paddle's surface angle in degrees.
        """
        return self.angle
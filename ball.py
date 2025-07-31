import pygame
import math
import numpy as np

import config

class Ball:
    """
    Represents the ball in the game, handling its physics, movement,
    and collision interactions with walls and paddles.
    """

    def __init__(self):
        """
        Initializes the ball object.
        Sets its initial position, size, and velocity. The initial serving
        direction is chosen randomly.
        """
        self.radius = config.BALL_RADIUS
        # Use float position for accurate physics
        self.x = config.SCREEN_WIDTH / 2
        self.y = config.SCREEN_HEIGHT / 2
        # The ball's position is managed by a Pygame Rect for easier collision detection
        # and drawing. The internal position is float-based for physics accuracy.
        self.rect = pygame.Rect(
            self.x - self.radius,
            self.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        # Start the first serve randomly
        self.reset(scored_left=np.random.choice([True, False]))

    def move(self, dt=1.0):
        """Updates the ball's position based on its current velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Update the rect position for collision detection
        self.rect.centerx = int(self.x)
        self.rect.centery = int(self.y)

    def handle_wall_collision(self):
        """
        Checks for and handles collisions with the top and bottom walls
        by reversing the vertical velocity.
        """
        if self.rect.top <= 0 or self.rect.bottom >= config.SCREEN_HEIGHT:
            self.vy *= -1
            # Clamp position to prevent the ball from getting stuck in a wall
            if self.rect.top < 0:
                self.rect.top = 0
                self.y = self.rect.centery
            if self.rect.bottom > config.SCREEN_HEIGHT:
                self.rect.bottom = config.SCREEN_HEIGHT
                self.y = self.rect.centery

    def handle_paddle_collision(self, paddle):
        """
        Handles collision with a paddle, calculating the new velocity based on
        the paddle's state (swinging or not).

        Args:
            paddle (Paddle): The paddle object the ball has collided with.
        """
        # This check prevents a "double hit" bug where the ball, after reversing
        # direction, is still inside the paddle and collides again. We only
        # process a collision if the ball is moving towards the paddle.
        is_moving_towards_paddle = (paddle.side == 'left' and self.vx < 0) or \
                                   (paddle.side == 'right' and self.vx > 0)
        if not is_moving_towards_paddle:
            return False

        if paddle.is_swinging:
            self._handle_swing_collision(paddle)
        else:
            self._handle_normal_collision(paddle)

        # After collision, push the ball out of the paddle's rect to prevent sticking.
        if paddle.side == 'left':
            self.rect.left = paddle.rect.right
            self.x = self.rect.centerx
        else: # right paddle
            self.rect.right = paddle.rect.left
            self.x = self.rect.centerx
        
        return True

    def increase_speed(self):
        """
        Gradually increases the x velocity to prevent the ball from getting stuck 
        bouncing vertically. Only applies a small boost when x velocity is too low
        relative to y velocity.
        """
        abs_vx = abs(self.vx)
        abs_vy = abs(self.vy)
        
        # Check if ball is moving too vertically
        is_too_vertical = (abs_vx < config.BALL_MIN_X_VELOCITY) or \
                         (abs_vx > 0 and abs_vy / abs_vx > config.BALL_VERTICAL_RATIO_THRESHOLD)
        
        if is_too_vertical:
            # Apply a small, constant boost in the current x direction
            if self.vx > 0:
                self.vx += config.BALL_X_BOOST_PER_FRAME
            elif self.vx < 0:
                self.vx -= config.BALL_X_BOOST_PER_FRAME
            else:
                # If vx is exactly 0, give it a small random direction
                import numpy as np
                self.vx = config.BALL_X_BOOST_PER_FRAME * (1 if np.random.random() > 0.5 else -1)

    def _handle_normal_collision(self, paddle):
        """
        Handles a standard, non-swinging collision, typical of classic Pong.
        The bounce angle is determined by where the ball hits the paddle.

        Args:
            paddle (Paddle): The paddle involved in the collision.
        """
        relative_y = self.rect.centery - paddle.rect.centery
        # Normalize the impact point from -1 (top) to 1 (bottom)
        normalized_y = relative_y / (paddle.height / 2)
        bounce_angle_deg = normalized_y * config.MAX_BOUNCE_ANGLE
        bounce_angle_rad = math.radians(bounce_angle_deg)

        direction = 1 if paddle.side == 'left' else -1

        # self.speed = config.BALL_SPEED_INITIAL
        self.vx = direction * self.speed * math.cos(bounce_angle_rad)
        self.vy = self.speed * math.sin(bounce_angle_rad)

    def _handle_swing_collision(self, paddle):
        """
        Handles a collision when the paddle is swinging, creating a pinball-like effect.
        The ball's exit velocity is calculated by reflecting its incoming velocity
        vector off the paddle's angled surface normal.

        Args:
            paddle (Paddle): The swinging paddle involved in the collision.
        """
        # Get paddle's surface orientation from the paddle object.
        paddle_surface_angle_rad = math.radians(paddle.get_surface_angle())

        # The normal vector is perpendicular to the paddle's surface.
        # For a surface at angle theta, its normal is at angle theta - 90 degrees.
        normal_angle_rad = paddle_surface_angle_rad - math.pi / 2

        # Create the normal vector using Pygame's Vector2 for reflection math
        normal_vector = pygame.math.Vector2(
            math.cos(normal_angle_rad),
            math.sin(normal_angle_rad)
        )
        incoming_velocity = pygame.math.Vector2(self.vx, self.vy)

        # Reflect the velocity vector across the surface normal
        reflected_velocity = incoming_velocity.reflect(normal_vector)

        # Apply the increased speed from the swing
        self.speed = config.BALL_SPEED_SWING
        if reflected_velocity.length() > 0:
            reflected_velocity.scale_to_length(self.speed)

        self.vx = reflected_velocity.x
        self.vy = reflected_velocity.y

    def check_score(self):
        """
        Checks if the ball has gone past a paddle, resulting in a score.

        Returns:
            str or None: 'right' if the right player scored (ball passed left side),
                         'left' if the left player scored (ball passed right side),
                         or None if no one has scored.
        """
        if self.rect.right < 0:
            return 'right'  # Right player scored
        if self.rect.left > config.SCREEN_WIDTH:
            return 'left'   # Left player scored
        return None

    def reset(self, scored_left):
        """
        Resets the ball to the center of the screen and sets a new velocity.
        The ball is served towards the player who just lost the point.

        Args:
            scored_left (bool): True if the left player scored, False otherwise.
        """
        self.x = config.SCREEN_WIDTH / 2
        self.y = config.SCREEN_HEIGHT / 2
        self.rect.center = (int(self.x), int(self.y))
        self.speed = config.BALL_SPEED_INITIAL

        # Ball moves towards the loser.
        # If left scored, ball goes right (positive direction).
        # If right scored, ball goes left (negative direction).
        direction = 1 if scored_left else -1
        
        # Serve at a random angle up to 45 degrees up or down
        angle = np.random.uniform(-math.pi / 4, math.pi / 4)

        self.vx = direction * self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)

    def draw(self, screen):
        """
        Draws the ball on the game screen.

        Args:
            screen (pygame.Surface): The Pygame surface to draw on.
        """
        pygame.draw.ellipse(screen, config.BALL_COLOR, self.rect)
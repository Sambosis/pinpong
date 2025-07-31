import pygame
import config
from ball import Ball
from paddle import Paddle

pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Test Basic Game")
clock = pygame.time.Clock()

# Create game objects
paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
ball = Ball()

print("Starting basic test...")
print(f"Ball position: {ball.rect.center}")
print(f"Ball velocity: {ball.vx}, {ball.vy}")
print(f"Paddle1 position: {paddle1.rect.center}")
print(f"Paddle2 position: {paddle2.rect.center}")

running = True
frame_count = 0

while running and frame_count < 300:  # Run for 5 seconds at 60fps
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Simple physics update
    ball.move(1.0 / 60.0)
    ball.handle_wall_collision()
    
    # Check for scoring
    scorer = ball.check_score()
    if scorer:
        print(f"Score! {scorer} player scored")
        ball.reset(scored_left=(scorer == 'right'))
    
    # Clear screen
    screen.fill(config.BLACK)
    
    # Draw objects
    paddle1.draw(screen)
    paddle2.draw(screen)
    ball.draw(screen)
    
    # Draw center line
    pygame.draw.line(screen, config.WHITE, (config.SCREEN_WIDTH // 2, 0), (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
    
    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

print("Test finished")
pygame.quit()
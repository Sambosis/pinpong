import pygame
import config
from ball import Ball

# Simple debug test for ball movement
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Ball Debug")
clock = pygame.time.Clock()

ball = Ball()

print("Ball Debug Info:")
print(f"Initial position: {ball.rect.center}")
print(f"Initial velocity: vx={ball.vx}, vy={ball.vy}")
print(f"Initial speed: {ball.speed}")
print(f"BALL_SPEED_INITIAL from config: {config.BALL_SPEED_INITIAL}")

running = True
frame = 0

while running and frame < 300:  # Run for 5 seconds
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update ball position
    dt = 1.0 / 60.0
    old_pos = ball.rect.center
    ball.move(dt)
    new_pos = ball.rect.center
    
    if frame % 60 == 0:  # Print every second
        print(f"Frame {frame}: pos=({new_pos[0]:.1f}, {new_pos[1]:.1f}), vel=({ball.vx:.2f}, {ball.vy:.2f})")
    
    ball.handle_wall_collision()
    
    # Draw
    screen.fill(config.BLACK)
    ball.draw(screen)
    
    # Draw trajectory line
    end_x = ball.rect.centerx + ball.vx * 10
    end_y = ball.rect.centery + ball.vy * 10
    pygame.draw.line(screen, config.WHITE, ball.rect.center, (end_x, end_y), 2)
    
    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()
print("Debug test completed")
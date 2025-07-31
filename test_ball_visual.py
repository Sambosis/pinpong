import pygame
import config
from ball import Ball

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Movement Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    ball = Ball()
    
    print(f"Ball initial position: {ball.x}, {ball.y}")
    print(f"Ball initial velocity: {ball.vx}, {ball.vy}")
    print(f"Ball rect position: {ball.rect.center}")
    
    running = True
    frame = 0

    while running and frame < 600:  # 10 seconds at 60fps
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Slow down the ball movement for visibility
        dt = 1.0 / 60.0
        ball.move(dt)
        ball.handle_wall_collision()
        
        # Print position every 30 frames (twice per second)
        if frame % 30 == 0:
            print(f"Frame {frame}: Ball at ({ball.x:.1f}, {ball.y:.1f}), rect at {ball.rect.center}")
        
        # Clear screen
        screen.fill(config.BLACK)
        
        # Draw ball
        ball.draw(screen)
        
        # Draw position info
        pos_text = font.render(f"Ball: ({ball.x:.0f}, {ball.y:.0f})", True, config.WHITE)
        screen.blit(pos_text, (10, 10))
        
        vel_text = font.render(f"Velocity: ({ball.vx:.2f}, {ball.vy:.2f})", True, config.WHITE)
        screen.blit(vel_text, (10, 50))
        
        frame_text = font.render(f"Frame: {frame}", True, config.WHITE)
        screen.blit(frame_text, (10, 90))
        
        # Draw trajectory line to see direction
        end_x = ball.x + ball.vx * 20
        end_y = ball.y + ball.vy * 20
        pygame.draw.line(screen, config.GRAY, (int(ball.x), int(ball.y)), (int(end_x), int(end_y)), 2)
        
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()
    print("Test completed")

if __name__ == '__main__':
    main()
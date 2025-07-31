import pygame
import config
from ball import Ball

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Simple Ball Movement Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)

    ball = Ball()
    
    print(f"Initial ball position: ({ball.x}, {ball.y})")
    print(f"Initial ball velocity: ({ball.vx}, {ball.vy})")
    print(f"Ball rect center: {ball.rect.center}")
    
    running = True
    frame = 0
    last_pos = (ball.x, ball.y)

    while running and frame < 300:  # 5 seconds at 60fps
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Move ball
        ball.move(1.0/60.0)
        ball.handle_wall_collision()
        
        # Check if position changed
        current_pos = (ball.x, ball.y)
        if current_pos != last_pos:
            print(f"Frame {frame}: Ball moved from {last_pos} to {current_pos}")
            last_pos = current_pos
        
        # Clear screen with different color
        screen.fill((50, 50, 50))  # Dark gray instead of black
        
        # Draw ball with larger size for visibility
        pygame.draw.circle(screen, (255, 255, 255), (int(ball.x), int(ball.y)), 15, 0)
        
        # Draw ball trail
        pygame.draw.circle(screen, (100, 100, 100), (int(ball.x), int(ball.y)), 25, 2)
        
        # Show position as text
        pos_text = font.render(f"({int(ball.x)}, {int(ball.y)})", True, (255, 255, 255))
        screen.blit(pos_text, (10, 10))
        
        # Show velocity as text
        vel_text = font.render(f"v: ({ball.vx:.2f}, {ball.vy:.2f})", True, (255, 255, 255))
        screen.blit(vel_text, (10, 60))
        
        # Show frame
        frame_text = font.render(f"Frame: {frame}", True, (255, 255, 255))
        screen.blit(frame_text, (10, 110))
        
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()
    print(f"Final ball position: ({ball.x}, {ball.y})")

if __name__ == '__main__':
    main()
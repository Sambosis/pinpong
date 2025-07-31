import pygame
import sys
import config
from ball import Ball
from paddle import Paddle

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Visual Pong Test")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Create game objects
    paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
    paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
    ball = Ball()
    
    score1 = 0
    score2 = 0
    
    print("Use WASD to control left paddle, Arrow keys for right paddle")
    print("Space to make left paddle swing, Right Shift for right paddle swing")
    print("Press ESC to quit")

    running = True
    dt = 1.0 / 60.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paddle1.swing()
                elif event.key == pygame.K_RSHIFT:
                    paddle2.swing()

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        
        # Left paddle controls (WASD)
        if keys[pygame.K_w]:
            paddle1.move(-1)
        elif keys[pygame.K_s]:
            paddle1.move(1)
            
        # Right paddle controls (Arrow keys)
        if keys[pygame.K_UP]:
            paddle2.move(-1)
        elif keys[pygame.K_DOWN]:
            paddle2.move(1)

        # Update physics
        paddle1.update(dt)
        paddle2.update(dt)
        ball.move(dt)
        ball.handle_wall_collision()
        
        # Check ball-paddle collisions
        if ball.rect.colliderect(paddle1.rect):
            ball.handle_paddle_collision(paddle1)
        if ball.rect.colliderect(paddle2.rect):
            ball.handle_paddle_collision(paddle2)
        
        # Check for scoring
        scorer = ball.check_score()
        if scorer:
            if scorer == 'left':
                score1 += 1
            else:
                score2 += 1
            print(f"Score! Current score: {score1} - {score2}")
            ball.reset(scored_left=(scorer == 'right'))

        # Render everything
        screen.fill(config.BLACK)
        
        # Draw paddles and ball
        paddle1.draw(screen)
        paddle2.draw(screen)
        ball.draw(screen)
        
        # Draw center line
        pygame.draw.aaline(screen, config.GRAY,
                           (config.SCREEN_WIDTH // 2, 0),
                           (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
        
        # Draw scores
        score_text = font.render(f"{score1}  -  {score2}", True, config.WHITE)
        screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))
        
        # Draw controls
        controls_text = font.render("WASD/Arrows to move, Space/RShift to swing", True, config.GRAY)
        screen.blit(controls_text, (10, config.SCREEN_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
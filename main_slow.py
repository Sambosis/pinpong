import pygame
import sys
import os
import time
import numpy as np

import config
from ball import Ball
from paddle import Paddle
from agent import Agent


def main():
    """
    Slow version where you can actually see the ball moving during AI training.
    """
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Slow AI Pong Training - Ball Movement Visible")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 20)

    print("Creating RL agents...")
    agent1 = Agent(agent_id='agent1')
    agent2 = Agent(agent_id='agent2')
    
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    total_wins_p1 = 0
    total_wins_p2 = 0
    running = True
    paused = False
    
    print("Starting SLOW visual training - you will see ball movement!")
    print("Controls: ESC to quit, P to pause/unpause")
    
    for episode in range(1, 20):  # Just 20 episodes for demo
        if not running:
            break
            
        print(f"Episode {episode} starting...")
        
        # Create game objects
        paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
        paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
        ball = Ball()
        
        score1 = 0
        score2 = 0
        episode_over = False
        step_count = 0
        
        while not episode_over and running and step_count < 1000:
            step_count += 1
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        episode_over = True
                    elif event.key == pygame.K_p:
                        paused = not paused

            if not running:
                break
                
            if paused:
                # Show pause screen
                screen.fill(config.BLACK)
                pause_text = font.render("PAUSED", True, config.WHITE)
                pause_rect = pause_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
                screen.blit(pause_text, pause_rect)
                pygame.display.flip()
                clock.tick(10)
                continue
            
            # Get game state
            ball_x_norm = ball.rect.centerx / config.SCREEN_WIDTH
            ball_y_norm = ball.rect.centery / config.SCREEN_HEIGHT
            ball_vx_norm = np.clip(ball.vx / config.BALL_MAX_SPEED, -1, 1)
            ball_vy_norm = np.clip(ball.vy / config.BALL_MAX_SPEED, -1, 1)
            p1_y_norm = paddle1.rect.centery / config.SCREEN_HEIGHT
            p2_y_norm = paddle2.rect.centery / config.SCREEN_HEIGHT
            p1_swing_norm = paddle1.swing_timer / config.PADDLE_SWING_DURATION if paddle1.swing_timer > 0 else 0
            p2_swing_norm = paddle2.swing_timer / config.PADDLE_SWING_DURATION if paddle2.swing_timer > 0 else 0
            
            state = np.array([
                ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm,
                p1_y_norm, p2_y_norm, p1_swing_norm, p2_swing_norm
            ], dtype=np.float32)
            
            # AI agents choose actions (only every few frames to slow things down)
            if step_count % 5 == 0:  # AI thinks every 5th frame
                action1 = agent1.choose_action(state)
                action2 = agent2.choose_action(state)
            else:
                action1 = 0  # STAY
                action2 = 0  # STAY
            
            # Apply actions to paddles
            if action1 == 1:
                paddle1.move(-1)  # UP
            elif action1 == 2:
                paddle1.move(1)   # DOWN
            elif action1 == 3:
                paddle1.swing()   # SWING
                
            if action2 == 1:
                paddle2.move(-1)  # UP
            elif action2 == 2:
                paddle2.move(1)   # DOWN
            elif action2 == 3:
                paddle2.swing()   # SWING

            # Update physics (every frame)
            dt = 1.0 / 15.0  # Slow physics timestep
            paddle1.update(dt)
            paddle2.update(dt)
            ball.move(dt)
            ball.handle_wall_collision()
            
            # Handle collisions
            reward1 = 0.0
            reward2 = 0.0
            
            if ball.rect.colliderect(paddle1.rect):
                if ball.handle_paddle_collision(paddle1):
                    reward1 = config.REWARD_HIT
                    print(f"P1 HIT at step {step_count}!")
                    
            if ball.rect.colliderect(paddle2.rect):
                if ball.handle_paddle_collision(paddle2):
                    reward2 = config.REWARD_HIT  
                    print(f"P2 HIT at step {step_count}!")
            
            # Check scoring
            scorer = ball.check_score()
            if scorer:
                if scorer == 'left':
                    score1 += 1
                    reward1 += config.REWARD_WIN
                    reward2 += config.REWARD_LOSE
                    print(f"P1 SCORED! {score1}-{score2}")
                else:
                    score2 += 1
                    reward1 += config.REWARD_LOSE
                    reward2 += config.REWARD_WIN
                    print(f"P2 SCORED! {score1}-{score2}")
                    
                ball.reset(scored_left=(scorer == 'right'))
                
                if score1 >= config.WINNING_SCORE or score2 >= config.WINNING_SCORE:
                    episode_over = True
                    if score1 > score2:
                        total_wins_p1 += 1
                    else:
                        total_wins_p2 += 1
                        
                    print(f"Episode {episode} finished: {score1}-{score2}")
            
            # Store experience and train (less frequently)
            if step_count % 10 == 0:  # Train every 10th step
                next_state = np.array([
                    ball.rect.centerx / config.SCREEN_WIDTH,
                    ball.rect.centery / config.SCREEN_HEIGHT,
                    np.clip(ball.vx / config.BALL_MAX_SPEED, -1, 1),
                    np.clip(ball.vy / config.BALL_MAX_SPEED, -1, 1),
                    paddle1.rect.centery / config.SCREEN_HEIGHT,
                    paddle2.rect.centery / config.SCREEN_HEIGHT,
                    paddle1.swing_timer / config.PADDLE_SWING_DURATION if paddle1.swing_timer > 0 else 0,
                    paddle2.swing_timer / config.PADDLE_SWING_DURATION if paddle2.swing_timer > 0 else 0
                ], dtype=np.float32)

                agent1.remember(state, action1, reward1, next_state, False)
                agent2.remember(state, action2, reward2, next_state, False)
                agent1.learn()
                agent2.learn()

            # Render the game (every frame for smooth visuals)
            screen.fill(config.BLACK)
            
            # Draw game elements
            paddle1.draw(screen)
            paddle2.draw(screen)
            ball.draw(screen)
            
            # Draw center line
            pygame.draw.aaline(screen, config.GRAY,
                               (config.SCREEN_WIDTH // 2, 0),
                               (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
            
            # Draw ball trail for visibility
            trail_length = 5
            for i in range(1, trail_length):
                trail_x = ball.x - (ball.vx * dt * i)
                trail_y = ball.y - (ball.vy * dt * i)
                alpha = 255 - (i * 50)
                if alpha > 0:
                    trail_color = (alpha, alpha, alpha)
                    pygame.draw.circle(screen, trail_color, (int(trail_x), int(trail_y)), ball.radius - i, 1)
            
            # Draw scores
            score_text = font.render(f"{score1}  -  {score2}", True, config.WHITE)
            screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 30))
            
            # Draw detailed info
            info_y = 10
            episode_text = small_font.render(f"Episode: {episode}/20", True, config.WHITE)
            screen.blit(episode_text, (10, info_y))
            info_y += 20
            
            step_text = small_font.render(f"Step: {step_count}", True, config.WHITE)
            screen.blit(step_text, (10, info_y))
            info_y += 20
            
            ball_text = small_font.render(f"Ball: ({ball.x:.0f},{ball.y:.0f})", True, config.WHITE)
            screen.blit(ball_text, (10, info_y))
            info_y += 20
            
            vel_text = small_font.render(f"Vel: ({ball.vx:.1f},{ball.vy:.1f})", True, config.WHITE)
            screen.blit(vel_text, (10, info_y))
            info_y += 20
            
            # Action indicators
            action_names = ["STAY", "UP", "DOWN", "SWING"]
            if step_count % 5 == 0:  # Only show when AI is thinking
                action1_text = small_font.render(f"P1: {action_names[action1]}", True, (255, 255, 0))
                action2_text = small_font.render(f"P2: {action_names[action2]}", True, (255, 255, 0))
            else:
                action1_text = small_font.render(f"P1: {action_names[0]}", True, config.GRAY)
                action2_text = small_font.render(f"P2: {action_names[0]}", True, config.GRAY)
                
            screen.blit(action1_text, (10, config.SCREEN_HEIGHT - 60))
            screen.blit(action2_text, (10, config.SCREEN_HEIGHT - 40))
            
            # Swing indicators
            if paddle1.is_swinging:
                swing_text = font.render("SWING!", True, (255, 255, 0))
                screen.blit(swing_text, (100, config.SCREEN_HEIGHT // 2))
                
            if paddle2.is_swinging:
                swing_text = font.render("SWING!", True, (255, 255, 0))
                swing_rect = swing_text.get_rect()
                swing_rect.right = config.SCREEN_WIDTH - 100
                swing_rect.centery = config.SCREEN_HEIGHT // 2
                screen.blit(swing_text, swing_rect)
            
            pygame.display.flip()
            clock.tick(15)  # Slow frame rate - 15 FPS

        # Post-episode processing
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if episode % 5 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated")

    print("Demo completed!")
    print(f"Final Results: P1 wins: {total_wins_p1}, P2 wins: {total_wins_p2}")
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
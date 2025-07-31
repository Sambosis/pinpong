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
    Fast version with proper ball speed so it actually reaches the paddles.
    """
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Fast AI Pong Training - Proper Speed")
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
    
    print("Starting training with proper ball speed!")
    print("Controls: ESC to quit, P to pause/unpause")
    
    for episode in range(1, 100):  # 100 episodes
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
        
        while not episode_over and running and step_count < 3000:  # Longer episode limit
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
            
            # AI agents choose actions every frame
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            
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

            # Update physics with proper timestep
            dt = 1.0 / 60.0  # Normal 60 FPS physics
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
                    print(f"P1 HIT! Ball speed: {abs(ball.vx):.1f}")
                    
            if ball.rect.colliderect(paddle2.rect):
                if ball.handle_paddle_collision(paddle2):
                    reward2 = config.REWARD_HIT  
                    print(f"P2 HIT! Ball speed: {abs(ball.vx):.1f}")
            
            # Check scoring
            scorer = ball.check_score()
            if scorer:
                if scorer == 'left':
                    score1 += 1
                    reward1 += config.REWARD_WIN
                    reward2 += config.REWARD_LOSE
                    print(f"P1 SCORED! {score1}-{score2} (step {step_count})")
                else:
                    score2 += 1
                    reward1 += config.REWARD_LOSE
                    reward2 += config.REWARD_WIN
                    print(f"P2 SCORED! {score1}-{score2} (step {step_count})")
                    
                ball.reset(scored_left=(scorer == 'right'))
                
                if score1 >= config.WINNING_SCORE or score2 >= config.WINNING_SCORE:
                    episode_over = True
                    if score1 > score2:
                        total_wins_p1 += 1
                    else:
                        total_wins_p2 += 1
                        
                    print(f"Episode {episode} finished: {score1}-{score2} in {step_count} steps")
            
            # Get next state
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

            # Store experience and train
            agent1.remember(state, action1, reward1, next_state, episode_over)
            agent2.remember(state, action2, reward2, next_state, episode_over)
            agent1.learn()
            agent2.learn()

            # Render the game
            screen.fill(config.BLACK)
            
            # Draw game elements
            paddle1.draw(screen)
            paddle2.draw(screen)
            ball.draw(screen)
            
            # Draw center line
            pygame.draw.aaline(screen, config.GRAY,
                               (config.SCREEN_WIDTH // 2, 0),
                               (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
            
            # Draw scores
            score_text = font.render(f"{score1}  -  {score2}", True, config.WHITE)
            screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 30))
            
            # Draw info
            info_y = 10
            episode_text = small_font.render(f"Episode: {episode}/100", True, config.WHITE)
            screen.blit(episode_text, (10, info_y))
            info_y += 20
            
            step_text = small_font.render(f"Step: {step_count}/3000", True, config.WHITE)
            screen.blit(step_text, (10, info_y))
            info_y += 20
            
            ball_text = small_font.render(f"Ball: ({ball.x:.0f},{ball.y:.0f})", True, config.WHITE)
            screen.blit(ball_text, (10, info_y))
            info_y += 20
            
            speed_text = small_font.render(f"Speed: {abs(ball.vx):.1f}", True, config.WHITE)
            screen.blit(speed_text, (10, info_y))
            info_y += 20
            
            eps_text = small_font.render(f"P1 eps: {agent1.epsilon:.3f}", True, config.WHITE)
            screen.blit(eps_text, (10, info_y))
            info_y += 18
            
            eps_text2 = small_font.render(f"P2 eps: {agent2.epsilon:.3f}", True, config.WHITE)
            screen.blit(eps_text2, (10, info_y))
            info_y += 18
            
            wins_text = small_font.render(f"Wins: {total_wins_p1}-{total_wins_p2}", True, config.WHITE)
            screen.blit(wins_text, (10, info_y))
            
            # Current actions
            action_names = ["STAY", "UP", "DOWN", "SWING"]
            action1_text = small_font.render(f"P1: {action_names[action1]}", True, config.WHITE)
            action2_text = small_font.render(f"P2: {action_names[action2]}", True, config.WHITE)
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
            clock.tick(60)  # 60 FPS for smooth visuals

        # Post-episode processing
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated")

        if episode % 10 == 0:
            print(f"Progress: Episode {episode} | Wins: P1={total_wins_p1}, P2={total_wins_p2} | Eps: {agent1.epsilon:.3f}")

    print("Training completed!")
    print(f"Final Results: P1 wins: {total_wins_p1}, P2 wins: {total_wins_p2}")
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
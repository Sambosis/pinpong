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
    Working version of the RL Pong game with proper display handling.
    """
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.GAME_CAPTION)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    print("Creating RL agents...")
    agent1 = Agent(agent_id='agent1')
    agent2 = Agent(agent_id='agent2')
    
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    total_wins_p1 = 0
    total_wins_p2 = 0
    running = True
    paused = False
    
    print(f"Starting training for {config.NUM_EPISODES} episodes...")
    print("Controls: ESC to quit, P to pause/unpause")
    
    start_time = time.time()
    
    for episode in range(1, config.NUM_EPISODES + 1):
        if not running:
            break
            
        print(f"Episode {episode} starting...")
        
        # Create game objects directly
        paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
        paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
        ball = Ball()
        
        score1 = 0
        score2 = 0
        episode_over = False
        step_count = 0
        
        while not episode_over and running and step_count < 2000:
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
                # Render pause state
                screen.fill(config.BLACK)
                pause_text = font.render("PAUSED - Press P to continue", True, config.WHITE)
                pause_rect = pause_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
                screen.blit(pause_text, pause_rect)
                pygame.display.flip()
                clock.tick(30)
                continue
            
            # Get game state for AI agents
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
            
            # AI agents choose actions
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)
            
            # Apply actions to paddles
            # Action mapping: 0: STAY, 1: UP, 2: DOWN, 3: SWING
            if action1 == 1:
                paddle1.move(-1)
            elif action1 == 2:
                paddle1.move(1)
            elif action1 == 3:
                paddle1.swing()
                
            if action2 == 1:
                paddle2.move(-1)
            elif action2 == 2:
                paddle2.move(1)
            elif action2 == 3:
                paddle2.swing()

            # Update physics
            dt = 1.0 / 60.0  # Slower for visibility
            paddle1.update(dt)
            paddle2.update(dt)
            ball.move(dt)
            ball.handle_wall_collision()
            
            # Handle ball-paddle collisions and calculate rewards
            reward1 = 0.0
            reward2 = 0.0
            
            if ball.rect.colliderect(paddle1.rect):
                if ball.handle_paddle_collision(paddle1):
                    reward1 = config.REWARD_HIT
                    
            if ball.rect.colliderect(paddle2.rect):
                if ball.handle_paddle_collision(paddle2):
                    reward2 = config.REWARD_HIT
            
            # Check for scoring
            scorer = ball.check_score()
            round_done = False
            if scorer:
                if scorer == 'left':
                    score1 += 1
                    reward1 += config.REWARD_WIN
                    reward2 += config.REWARD_LOSE
                else:
                    score2 += 1
                    reward1 += config.REWARD_LOSE
                    reward2 += config.REWARD_WIN
                    
                ball.reset(scored_left=(scorer == 'right'))
                
                if score1 >= config.WINNING_SCORE or score2 >= config.WINNING_SCORE:
                    episode_over = True
                    round_done = True
                    if score1 > score2:
                        total_wins_p1 += 1
                    else:
                        total_wins_p2 += 1
                        
                    print(f"Episode {episode} finished: {score1} - {score2}")
            
            # Get next state
            next_ball_x_norm = ball.rect.centerx / config.SCREEN_WIDTH
            next_ball_y_norm = ball.rect.centery / config.SCREEN_HEIGHT
            next_ball_vx_norm = np.clip(ball.vx / config.BALL_MAX_SPEED, -1, 1)
            next_ball_vy_norm = np.clip(ball.vy / config.BALL_MAX_SPEED, -1, 1)
            next_p1_y_norm = paddle1.rect.centery / config.SCREEN_HEIGHT
            next_p2_y_norm = paddle2.rect.centery / config.SCREEN_HEIGHT
            next_p1_swing_norm = paddle1.swing_timer / config.PADDLE_SWING_DURATION if paddle1.swing_timer > 0 else 0
            next_p2_swing_norm = paddle2.swing_timer / config.PADDLE_SWING_DURATION if paddle2.swing_timer > 0 else 0
            
            next_state = np.array([
                next_ball_x_norm, next_ball_y_norm, next_ball_vx_norm, next_ball_vy_norm,
                next_p1_y_norm, next_p2_y_norm, next_p1_swing_norm, next_p2_swing_norm
            ], dtype=np.float32)

            # Store experiences and train
            agent1.remember(state, action1, reward1, next_state, round_done)
            agent2.remember(state, action2, reward2, next_state, round_done)
            agent1.learn()
            agent2.learn()

            # Render every frame during training
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
            screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))
            
            # Draw episode and training info
            info_y = 10
            episode_text = small_font.render(f"Episode: {episode}/{config.NUM_EPISODES}", True, config.WHITE)
            screen.blit(episode_text, (10, info_y))
            info_y += 25
            
            eps_text = small_font.render(f"P1 eps: {agent1.epsilon:.4f}", True, config.WHITE)
            screen.blit(eps_text, (10, info_y))
            info_y += 20
            
            eps_text2 = small_font.render(f"P2 eps: {agent2.epsilon:.4f}", True, config.WHITE)
            screen.blit(eps_text2, (10, info_y))
            info_y += 20
            
            wins_text = small_font.render(f"Wins: P1={total_wins_p1}, P2={total_wins_p2}", True, config.WHITE)
            screen.blit(wins_text, (10, info_y))
            
            # Draw swing indicators
            if paddle1.is_swinging:
                swing_text = small_font.render("P1 SWING!", True, config.WHITE)
                screen.blit(swing_text, (50, config.SCREEN_HEIGHT // 2 - 50))
                
            if paddle2.is_swinging:
                swing_text = small_font.render("P2 SWING!", True, config.WHITE)
                swing_rect = swing_text.get_rect()
                swing_rect.right = config.SCREEN_WIDTH - 50
                swing_rect.centery = config.SCREEN_HEIGHT // 2 - 50
                screen.blit(swing_text, swing_rect)
            
            pygame.display.flip()
            clock.tick(30)  # Slower frame rate for better visibility

        # Post-episode processing
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Update target networks periodically
        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated")

        if episode % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Wins: P1={total_wins_p1}, P2={total_wins_p2} | "
                  f"Epsilon: P1={agent1.epsilon:.4f}, P2={agent2.epsilon:.4f}")
            start_time = time.time()

    print("Training completed!")
    print(f"Final Results: P1 wins: {total_wins_p1}, P2 wins: {total_wins_p2}")
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
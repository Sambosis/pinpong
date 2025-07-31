import pygame
import sys
import os
import time
import numpy as np

import config
from game import PongGame  
from agent import Agent


def main():
    """
    Final complete version of the RL Pong game with pinball flipper mechanics.
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

    # Training statistics
    total_wins_p1 = 0
    total_wins_p2 = 0
    episode_lengths = []
    running = True
    paused = False
    
    print(f"Starting training for {config.NUM_EPISODES} episodes...")
    print("Controls: ESC to quit, P to pause/unpause, R to reset epsilon")
    
    start_time = time.time()
    
    for episode in range(1, config.NUM_EPISODES + 1):
        if not running:
            break
            
        # Decide whether to render this episode
        should_render = (episode % config.DISPLAY_EVERY == 0)
        
        # Create game environment
        game_env = PongGame(render_mode=should_render)
        
        # Reset the game
        state = game_env.reset()
        episode_over = False
        step_count = 0
        episode_start_time = time.time()
        
        while not episode_over and running and step_count < 2000:  # Prevent infinite episodes
            step_count += 1
            
            # Handle pygame events only when rendering
            if should_render:
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
                            print(f"Game {'paused' if paused else 'resumed'}")
                        elif event.key == pygame.K_r:
                            agent1.epsilon = config.EPSILON_START * 0.5
                            agent2.epsilon = config.EPSILON_START * 0.5
                            print("Epsilon reset for more exploration")

            if not running or (paused and should_render):
                if should_render:
                    # Still render when paused
                    screen.fill(config.BLACK)
                    
                    # Draw game elements
                    game_env.paddle1.draw(screen)
                    game_env.paddle2.draw(screen)
                    game_env.ball.draw(screen)
                    
                    # Draw center line
                    pygame.draw.aaline(screen, config.GRAY,
                                       (config.SCREEN_WIDTH // 2, 0),
                                       (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
                    
                    # Draw pause indicator
                    pause_text = font.render("PAUSED - Press P to continue", True, config.WHITE)
                    pause_rect = pause_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
                    screen.blit(pause_text, pause_rect)
                    
                    pygame.display.flip()
                    clock.tick(30)
                continue
                
            # AI agents choose actions
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)

            # Execute actions in the environment
            dt = 1.0 / 120.0  # High frequency physics
            next_state, reward1, reward2, round_done = game_env.step(action1, action2, dt)

            # Store experiences for learning
            agent1.remember(state, action1, reward1, next_state, round_done)
            agent2.remember(state, action2, reward2, next_state, round_done)

            # Update current state
            state = next_state

            # Train the neural networks
            agent1.learn()
            agent2.learn()
            
            # Check if game/episode ended
            if round_done or game_env.score1 >= config.WINNING_SCORE or game_env.score2 >= config.WINNING_SCORE:
                episode_over = True
                episode_length = time.time() - episode_start_time
                episode_lengths.append(episode_length)
                
                # Update win statistics
                if game_env.score1 > game_env.score2:
                    total_wins_p1 += 1
                elif game_env.score2 > game_env.score1:
                    total_wins_p2 += 1

            # Render the game if this episode should be displayed
            if should_render:
                screen.fill(config.BLACK)
                
                # Draw game elements
                game_env.paddle1.draw(screen)
                game_env.paddle2.draw(screen)
                game_env.ball.draw(screen)
                
                # Draw center line
                pygame.draw.aaline(screen, config.GRAY,
                                   (config.SCREEN_WIDTH // 2, 0),
                                   (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))
                
                # Draw scores
                score_text = font.render(f"{game_env.score1}  -  {game_env.score2}", True, config.WHITE)
                screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))
                
                # Draw episode and training info
                info_y = 10
                episode_text = small_font.render(f"Episode: {episode}/{config.NUM_EPISODES}", True, config.WHITE)
                screen.blit(episode_text, (10, info_y))
                info_y += 25
                
                eps_text = small_font.render(f"P1 epsilon: {agent1.epsilon:.4f}", True, config.WHITE)
                screen.blit(eps_text, (10, info_y))
                info_y += 20
                
                eps_text2 = small_font.render(f"P2 epsilon: {agent2.epsilon:.4f}", True, config.WHITE)
                screen.blit(eps_text2, (10, info_y))
                info_y += 20
                
                wins_text = small_font.render(f"Wins: P1={total_wins_p1}, P2={total_wins_p2}", True, config.WHITE)
                screen.blit(wins_text, (10, info_y))
                info_y += 20
                
                if len(episode_lengths) > 0:
                    avg_length = np.mean(episode_lengths[-10:])  # Average of last 10 episodes
                    length_text = small_font.render(f"Avg episode: {avg_length:.1f}s", True, config.WHITE)
                    screen.blit(length_text, (10, info_y))
                
                # Draw swing indicators
                if game_env.paddle1.is_swinging:
                    swing_text = small_font.render("P1 SWING!", True, config.WHITE)
                    screen.blit(swing_text, (50, config.SCREEN_HEIGHT // 2))
                    
                if game_env.paddle2.is_swinging:
                    swing_text = small_font.render("P2 SWING!", True, config.WHITE)
                    swing_rect = swing_text.get_rect()
                    swing_rect.right = config.SCREEN_WIDTH - 50
                    swing_rect.centery = config.SCREEN_HEIGHT // 2
                    screen.blit(swing_text, swing_rect)
                
                # Draw controls
                controls_text = small_font.render("ESC: quit | P: pause | R: reset epsilon", True, config.GRAY)
                screen.blit(controls_text, (10, config.SCREEN_HEIGHT - 25))
                
                pygame.display.flip()
                clock.tick(config.FPS)

        # Close the game environment
        game_env.close()

        # Post-episode processing
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Periodically update target networks
        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated")

        # Periodically save models
        if episode % config.SAVE_MODEL_EVERY == 0:
            agent1.save_model(episode)
            agent2.save_model(episode)
            print(f"Models saved at episode {episode}")

        # Print progress summary
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_episode_time = elapsed_time / 10 if episode > 0 else 0
            
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Wins: P1={total_wins_p1}, P2={total_wins_p2} | "
                  f"Last Score: {game_env.score1}-{game_env.score2} | "
                  f"Epsilon: P1={agent1.epsilon:.4f}, P2={agent2.epsilon:.4f} | "
                  f"Time/ep: {avg_episode_time:.2f}s")
            
            start_time = time.time()

    print("Training completed!")
    print(f"Final Results: P1 wins: {total_wins_p1}, P2 wins: {total_wins_p2}")
    
    # Save final models
    agent1.save_model(config.NUM_EPISODES)
    agent2.save_model(config.NUM_EPISODES)
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
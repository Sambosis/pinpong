import pygame
import sys
import os
import numpy as np

import config
from game import PongGame  
from agent import Agent


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.GAME_CAPTION)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    print("Creating agents...")
    agent1 = Agent(agent_id='agent1')
    agent2 = Agent(agent_id='agent2')
    
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    total_wins_p1 = 0
    total_wins_p2 = 0
    running = True
    
    print(f"Starting training for {config.NUM_EPISODES} episodes...")
    
    for episode in range(1, config.NUM_EPISODES + 1):
        if not running:
            break
            
        print(f"Starting episode {episode}")
        
        # Create game (always with rendering for simplicity)
        game_env = PongGame(render_mode=True)
        
        # Reset the game
        state = game_env.reset()
        episode_over = False
        step_count = 0
        
        while not episode_over and running and step_count < 1000:
            step_count += 1
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        episode_over = True

            if not running:
                break
                
            # Agents choose actions
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)

            # Execute actions and get results
            dt = 1.0 / 60.0
            next_state, reward1, reward2, round_done = game_env.step(action1, action2, dt)

            # Store experiences
            agent1.remember(state, action1, reward1, next_state, round_done)
            agent2.remember(state, action2, reward2, next_state, round_done)

            # Update state
            state = next_state

            # Train agents
            agent1.learn()
            agent2.learn()
            
            # Check if game ended
            if round_done or game_env.score1 >= config.WINNING_SCORE or game_env.score2 >= config.WINNING_SCORE:
                episode_over = True
                if game_env.score1 > game_env.score2:
                    total_wins_p1 += 1
                else:
                    total_wins_p2 += 1
                    
                print(f"Episode {episode} finished: Score {game_env.score1} - {game_env.score2}")

            # Render the game
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
            screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))
            
            # Draw episode info
            episode_text = font.render(f"Episode: {episode}", True, config.WHITE)
            screen.blit(episode_text, (10, 10))
            
            # Draw agent info
            eps_text = font.render(f"P1 eps: {agent1.epsilon:.3f} | P2 eps: {agent2.epsilon:.3f}", True, config.WHITE)
            screen.blit(eps_text, (10, 40))
            
            wins_text = font.render(f"Wins: {total_wins_p1} - {total_wins_p2}", True, config.WHITE)
            screen.blit(wins_text, (10, 70))
            
            pygame.display.flip()
            clock.tick(config.FPS)

        # Close the game window
        game_env.close()

        # Post-episode tasks
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated.")

        if episode % 5 == 0:
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Total Wins: P1 {total_wins_p1} - P2 {total_wins_p2} | "
                  f"Epsilon P1: {agent1.epsilon:.4f} | Epsilon P2: {agent2.epsilon:.4f}")

    print("Training finished.")
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
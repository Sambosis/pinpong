import pygame
import sys
import os
import time
import numpy as np

# Internal project imports
import config
from game import PongGame  
from agent import Agent


def main():
    """
    The main entry point for the application.
    Initializes Pygame, the game environment, and the RL agents.
    Runs the main training loop, handles user input, rendering, and model management.
    """
    # --- Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.GAME_CAPTION)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    # Create the game environment and the two competing agents
    agent1 = Agent(agent_id='agent1')
    agent2 = Agent(agent_id='agent2')

    # Ensure the directory for saving models exists
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    # Load pre-trained models if configured to do so
    if config.LOAD_MODEL:
        if os.path.exists(config.MODEL_TO_LOAD_P1):
            agent1.load_model(config.MODEL_TO_LOAD_P1)
        if os.path.exists(config.MODEL_TO_LOAD_P2):
            agent2.load_model(config.MODEL_TO_LOAD_P2)

    total_wins_p1 = 0
    total_wins_p2 = 0
    paused = False
    running = True

    # --- Main Training Loop ---
    start_time = time.time()
    for episode in range(1, config.NUM_EPISODES + 1):
        if not running:
            break

        # Create game environment for this episode  
        render_this_episode = (episode % config.DISPLAY_EVERY == 0)
        game_env = PongGame(render_mode=render_this_episode)
        
        # Reset the game environment
        state = game_env.reset()
        episode_over = False
        last_time = time.time()

        while not episode_over:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # --- Event Handling (Quit, Pause) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_p:
                        paused = not paused

            if not running:
                episode_over = True
                continue

            # If paused, freeze the game logic but keep rendering
            if paused and render_this_episode:
                game_env.render(episode)
                # Draw pause indicator
                pause_text = font.render("PAUSED - Press P to resume", True, config.WHITE)
                pause_rect = pause_text.get_rect(center=(config.SCREEN_WIDTH / 2, config.SCREEN_HEIGHT / 2))
                screen.blit(pause_text, pause_rect)
                pygame.display.flip()
                clock.tick(config.FPS)
                continue

            # --- Agent-Environment Interaction ---
            # 1. Agents choose actions based on the current state
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)

            # 2. The environment executes the actions and returns the outcome
            next_state, reward1, reward2, round_done = game_env.step(action1, action2, dt)

            # 3. Agents store this transition in their replay memory
            agent1.remember(state, action1, reward1, next_state, round_done)
            agent2.remember(state, action2, reward2, next_state, round_done)

            # 4. Update the current state for the next iteration
            state = next_state

            # 5. Train the agents' networks by sampling from memory
            agent1.learn()
            agent2.learn()
            
            # 6. Check if the game (episode) has ended
            if round_done or game_env.score1 >= config.WINNING_SCORE or game_env.score2 >= config.WINNING_SCORE:
                episode_over = True
                if game_env.score1 > game_env.score2:
                    total_wins_p1 += 1
                else:
                    total_wins_p2 += 1

            # --- Rendering (conditional) ---
            # Only render every Nth game to speed up training
            if render_this_episode:
                screen.fill(config.BLACK)
                game_env.render(episode)
                
                # Display training stats
                eps_text1 = font.render(f"P1 eps: {agent1.epsilon:.3f}", True, config.WHITE)
                eps_text2 = font.render(f"P2 eps: {agent2.epsilon:.3f}", True, config.WHITE)
                episode_text = font.render(f"Episode: {episode}", True, config.WHITE)
                wins_text = font.render(f"Wins: {total_wins_p1} - {total_wins_p2}", True, config.WHITE)
                
                screen.blit(episode_text, (10, 10))
                screen.blit(eps_text1, (10, 40))
                screen.blit(eps_text2, (10, 70))
                screen.blit(wins_text, (10, 100))
                
                pygame.display.flip()
                clock.tick(config.FPS) # Control frame rate only when rendering

        # Close the game window
        game_env.close()

        # --- Post-Episode Tasks ---
        # Decay epsilon to shift from exploration to exploitation
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Periodically update the target networks for training stability
        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated.")

        # Periodically save the trained models
        if episode % config.SAVE_MODEL_EVERY == 0:
            agent1.save_model(episode)
            agent2.save_model(episode)

        # Print a summary of training progress to the console
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Total Wins: P1 {total_wins_p1} - P2 {total_wins_p2} | "
                  f"Last Score: {game_env.score1} - {game_env.score2} | "
                  f"Epsilon P1: {agent1.epsilon:.4f} | Epsilon P2: {agent2.epsilon:.4f} | "
                  f"Time/10eps: {elapsed_time:.2f}s")
            start_time = time.time()  # Reset timer for the next block of episodes

    # --- Cleanup ---
    print("Training finished.")
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
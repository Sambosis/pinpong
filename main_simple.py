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
    Simplified main function for testing and debugging.
    """
    print("Initializing pygame...")
    pygame.init()
    
    print("Creating agents...")
    agent1 = Agent(agent_id='agent1')
    agent2 = Agent(agent_id='agent2')
    
    print("Ensuring models directory exists...")
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    total_wins_p1 = 0
    total_wins_p2 = 0
    
    print(f"Starting training for {config.NUM_EPISODES} episodes...")
    
    for episode in range(1, config.NUM_EPISODES + 1):
        print(f"Starting episode {episode}")
        
        # Create game environment for this episode  
        render_this_episode = (episode % config.DISPLAY_EVERY == 0)
        game_env = PongGame(render_mode=render_this_episode)
        
        # Reset the game environment
        state = game_env.reset()
        episode_over = False
        
        step_count = 0
        while not episode_over and step_count < 1000:  # Add step limit to prevent infinite loops
            step_count += 1
            
            # Agents choose actions based on the current state
            action1 = agent1.choose_action(state)
            action2 = agent2.choose_action(state)

            # Execute actions and get results
            dt = 1.0 / 60.0  # Fixed timestep
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
                    
                print(f"Episode {episode} finished: {game_env.score1} - {game_env.score2}")

            # Render if needed
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                        
                game_env.render(episode)

        # Close the game window
        game_env.close()

        # Post-episode tasks
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Update target networks periodically
        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Target networks updated.")

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{config.NUM_EPISODES} | "
                  f"Total Wins: P1 {total_wins_p1} - P2 {total_wins_p2} | "
                  f"Epsilon P1: {agent1.epsilon:.4f} | Epsilon P2: {agent2.epsilon:.4f}")

    print("Training finished.")
    pygame.quit()


if __name__ == '__main__':
    main()
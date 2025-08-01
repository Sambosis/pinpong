import pygame
import sys
import os
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


import config
from ball import Ball
from paddle import Paddle
from agent import Agent
from vtrace_agent import VTraceAgent


def main():
    """
    Visual version with forced longer episodes and debug info.
    """
    # Initialize rich console
    console = Console()

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(config.GAME_CAPTION + " - Visual Training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 20)

    console.print(Panel.fit("🤖 Creating RL agents...", style="bold blue"))
    agent1 = Agent(agent_id='agent1')
    agent2 = VTraceAgent(agent_id='agent2')

    # Load pre-trained models if enabled
    if config.LOAD_MODEL:
        console.print(Panel.fit("📥 Loading pre-trained models...", style="bold cyan"))

        # Load Player 1 model
        if os.path.exists(config.MODEL_TO_LOAD_P1):
            agent1.load_model(config.MODEL_TO_LOAD_P1)
            console.print(f"[green]✅ Player 1 model loaded from:[/green] {config.MODEL_TO_LOAD_P1}")
        else:
            console.print(f"[red]❌ Player 1 model not found:[/red] {config.MODEL_TO_LOAD_P1}")
            console.print("[yellow]⚠️  Player 1 will start training from scratch[/yellow]")

        # Load Player 2 model
        if os.path.exists(config.MODEL_TO_LOAD_P2):
            agent2.load_model(config.MODEL_TO_LOAD_P2)
            console.print(f"[green]✅ Player 2 model loaded from:[/green] {config.MODEL_TO_LOAD_P2}")
        else:
            console.print(f"[red]❌ Player 2 model not found:[/red] {config.MODEL_TO_LOAD_P2}")
            console.print("[yellow]⚠️  Player 2 will start training from scratch[/yellow]")

        console.print()  # Add spacing

    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    total_wins_p1 = 0
    total_wins_p2 = 0
    running = True
    paused = False

    # Create training info table
    training_table = Table(title="🏓 Training Configuration")
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="magenta")
    training_table.add_row("Total Episodes", str(config.NUM_EPISODES))
    training_table.add_row("Display Every", f"{config.DISPLAY_EVERY} episodes")
    training_table.add_row("Winning Score", str(config.WINNING_SCORE))
    training_table.add_row("Ball Speed", f"{config.BALL_SPEED_INITIAL}")
    training_table.add_row("Load Models", "Yes" if config.LOAD_MODEL else "No")
    training_table.add_row("Save Every", f"{config.SAVE_MODEL_EVERY} episodes")

    console.print(training_table)
    console.print("\n[bold yellow]Controls:[/bold yellow] [green]ESC[/green] to quit, [green]P[/green] to pause/unpause, [green]SPACE[/green] to skip episode\n")

    for episode in range(1, config.NUM_EPISODES + 1):
        if not running:
            break

        # Decide whether to render this episode
        should_render = (episode % config.DISPLAY_EVERY == 0)

        if should_render:
            console.print(f"[bold green]📺 Episode {episode} starting (VISUAL)...[/bold green]")
        else:
            console.print(f"[dim]Episode {episode} starting...[/dim]")

        # Create game objects
        paddle1 = Paddle(config.PADDLE_WALL_OFFSET, 'left')
        paddle2 = Paddle(config.SCREEN_WIDTH - config.PADDLE_WIDTH - config.PADDLE_WALL_OFFSET, 'right')
        ball = Ball()

        score1 = 0
        score2 = 0
        episode_over = False
        step_count = 0
        episode_start_time = time.time()

        # Track hits, swings, and rewards for this episode
        hits_p1 = 0
        hits_p2 = 0
        swings_p1 = 0
        swings_p2 = 0
        total_reward_p1 = 0.0
        total_reward_p2 = 0.0
        cumulative_total_avg_reward = 0.0  # For calculating cumulative average reward
        while not episode_over and running and step_count < config.MAX_STEPS_PER_EPISODE:
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
                            status = "[red]⏸️  Paused[/red]" if paused else "[green]▶️  Resumed[/green]"
                            console.print(status)
                        elif event.key == pygame.K_SPACE:
                            episode_over = True

            if not running:
                break

            if paused and should_render:
                # Render pause state
                screen.fill(config.BLACK)
                pause_text = font.render("PAUSED - Press P to continue", True, config.WHITE)
                pause_rect = pause_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2))
                screen.blit(pause_text, pause_rect)
                pygame.display.flip()
                clock.tick(10)
                continue

            # Get normalized game state
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

            # Update physics
            dt = config.FIXED_TIMESTEP  # Use fixed timestep for physics updates
            paddle1.update(dt)
            paddle2.update(dt)
            ball.move(dt)
            ball.handle_wall_collision()
            ball.increase_speed()  # Apply anti-stall mechanism

            # Handle ball-paddle collisions and apply swing penalties
            reward1 = 0.0
            reward2 = 0.0

            # Apply swing penalties and count swings
            if action1 == 3:
                reward1 += config.REWARD_SWING
                swings_p1 += 1
            if action2 == 3:
                reward2 += config.REWARD_SWING
                swings_p2 += 1

            if ball.rect.colliderect(paddle1.rect):
                if ball.handle_paddle_collision(paddle1):
                    reward1 += config.REWARD_HIT
                    hits_p1 += 1

            if ball.rect.colliderect(paddle2.rect):
                if ball.handle_paddle_collision(paddle2):
                    reward2 += config.REWARD_HIT  
                    hits_p2 += 1

            # Check for scoring
            scorer = ball.check_score()
            round_done = False
            if scorer:
                if should_render:
                    # pause for 1 second to show the score
                    pygame.time.delay(1000)
                if scorer == 'left':
                    score1 += 1
                    reward1 += config.REWARD_WIN
                    reward2 += config.REWARD_LOSE
                else:
                    score2 += 1
                    reward1 += config.REWARD_LOSE
                    reward2 += config.REWARD_WIN

                ball.reset(scored_left=(scorer == 'right'))

                # End episode immediately if someone reached winning score
                if score1 >= config.WINNING_SCORE or score2 >= config.WINNING_SCORE:
                    episode_over = True
                    round_done = True

                    # Add game win/lose rewards
                    if score1 >= config.WINNING_SCORE:
                        reward1 += config.REWARD_GAME_WIN
                        reward2 += config.REWARD_GAME_LOSE
                        total_wins_p1 += 1
                    else:
                        reward1 += config.REWARD_GAME_LOSE
                        reward2 += config.REWARD_GAME_WIN
                        total_wins_p2 += 1

                    duration = time.time() - episode_start_time
                    total_hits = hits_p1 + hits_p2
                    total_swings = swings_p1 + swings_p2
                    avg_reward_p1 = total_reward_p1 / step_count if step_count > 0 else 0
                    avg_reward_p2 = total_reward_p2 / step_count if step_count > 0 else 0
                    total_avg_reward = avg_reward_p1 + avg_reward_p2
                    # calculate and save the cumulative total average reward
                    cumulative_total_avg_reward = (cumulative_total_avg_reward * (episode - 1) + total_avg_reward) / episode

                    # Create episode summary table
                    summary_table = Table(show_header=False, box=None, padding=(0, 1))
                    summary_table.add_column("", style="cyan")
                    summary_table.add_column("", style="white")

                    summary_table.add_row("🥊 Swing %", f"{total_swings / (step_count * 2) * 100:.3f}%")
                    summary_table.add_row("🏓 Hits / Swing", f"{total_hits / total_swings:.3f}" if total_swings > 0 else "0.000")
                    summary_table.add_row("📈 P1%  CTR", f"{avg_reward_p1 / cumulative_total_avg_reward:.3f}" if cumulative_total_avg_reward > 0 else "0.000")
                    summary_table.add_row(
                        f"📈 P2% of {cumulative_total_avg_reward:.3f}",
                        (
                            f"{avg_reward_p2 / cumulative_total_avg_reward:.3f}"
                            if cumulative_total_avg_reward > 0
                            else "0.000"
                        ),
                    )
                    summary_table.add_row("🏆 Total Wins", f"P1: {total_wins_p1} | P2: {total_wins_p2}")

                    console.print(summary_table)

            # Accumulate total rewards for this episode
            total_reward_p1 += reward1
            total_reward_p2 += reward2

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

            # Store experiences and train
            agent1.remember(state, action1, reward1, next_state, round_done)
            agent2.remember(state, action2, reward2, next_state, round_done)

            agent1.learn()
            agent2.learn()

            # Render the game only if this episode should be displayed
            if should_render:
                screen.fill(config.BLACK)

                # Draw game elements
                paddle1.draw(screen)
                paddle2.draw(screen)
                ball.draw(screen)

                # Draw center line
                pygame.draw.aaline(screen, config.GRAY,
                                   (config.SCREEN_WIDTH // 2, 0),
                                   (config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT))

                # Draw scores (large)
                score_text = font.render(f"{score1}  -  {score2}", True, config.WHITE)
                screen.blit(score_text, (config.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 30))

                # Draw info panel
                info_y = 10
                episode_text = small_font.render(f"Episode: {episode}/{config.NUM_EPISODES}", True, config.WHITE)
                screen.blit(episode_text, (10, info_y))
                info_y += 22

                step_text = small_font.render(f"Step: {step_count}/{config.MAX_STEPS_PER_EPISODE}", True, config.WHITE)
                screen.blit(step_text, (10, info_y))
                info_y += 22


                # Calculate and display vertical ratio (same logic as in ball.py)
                abs_vx = abs(ball.vx)
                abs_vy = abs(ball.vy)
                vertical_ratio = abs_vy / abs_vx if abs_vx > 0 else float('inf')
                ratio_color = config.RED if vertical_ratio > config.BALL_VERTICAL_RATIO_THRESHOLD else config.WHITE
                ratio_text = small_font.render(f"Vertical Ratio: {vertical_ratio:.2f} (threshold: {config.BALL_VERTICAL_RATIO_THRESHOLD})", True, ratio_color)
                screen.blit(ratio_text, (10, info_y))
                info_y += 22

                # Show anti-stall status (same logic as in ball.py increase_speed method)
                is_too_vertical = (abs_vx < config.BALL_MIN_X_VELOCITY) or \
                                 (abs_vx > 0 and abs_vy / abs_vx > config.BALL_VERTICAL_RATIO_THRESHOLD)
                stall_status = "ACTIVE" if is_too_vertical else "INACTIVE"
                stall_color = config.GREEN if is_too_vertical else config.GRAY
                stall_text = small_font.render(f"Anti-stall: {stall_status} (min_vx: {config.BALL_MIN_X_VELOCITY})", True, stall_color)
                screen.blit(stall_text, (10, info_y))
                info_y += 22

                eps_text = small_font.render(f"Epsilon: {agent1.epsilon:.3f}", True, config.WHITE)
                screen.blit(eps_text, (10, info_y))
                info_y += 20



                wins_text = small_font.render(f"Wins: P1={total_wins_p1}, P2={total_wins_p2}", True, config.WHITE)
                screen.blit(wins_text, (10, info_y))

                # Action display
                action_names = ["STAY", "UP", "DOWN", "SWING"]
                # action1_text = small_font.render(f"P1: {action_names[action1]}", True, config.WHITE)
                # screen.blit(action1_text, (10, config.SCREEN_HEIGHT - 60))

                # action2_text = small_font.render(f"P2: {action_names[action2]}", True, config.WHITE)
                # screen.blit(action2_text, (10, config.SCREEN_HEIGHT - 40))
                


                pygame.display.flip()
                clock.tick(config.FPS)  # 60 FPS

        # Post-episode processing
        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Save models periodically
        if episode % config.SAVE_MODEL_EVERY == 0:
            agent1.save_model(episode)
            agent2.save_model(episode)
            console.print(f"[bold green]💾 Episode {episode}: Models saved[/bold green]")

        # Update target networks
        if episode % config.TARGET_UPDATE_FREQUENCY == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            console.print(f"[bold yellow]🔄 Episode {episode}: Target networks updated[/bold yellow]")

    # Training completion summary
    final_table = Table(title="🎯 Training Completed!")
    final_table.add_column("Player", style="cyan")
    final_table.add_column("Wins", style="magenta")
    final_table.add_column("Win Rate", style="green")

    total_games = total_wins_p1 + total_wins_p2
    p1_rate = (total_wins_p1 / total_games * 100) if total_games > 0 else 0
    p2_rate = (total_wins_p2 / total_games * 100) if total_games > 0 else 0

    final_table.add_row("Player 1", str(total_wins_p1), f"{p1_rate:.1f}%")
    final_table.add_row("Player 2", str(total_wins_p2), f"{p2_rate:.1f}%")

    console.print(final_table)
    console.print("[bold green]🎉 Training session complete![/bold green]")

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()

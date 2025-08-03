import pygame
import sys
import os
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import imageio
import pygame
import numpy as np
import os
import math
import random


import config
from ball import Ball
from paddle import Paddle
from agent import Agent
from vtrace_agent import VTraceAgent


class Recorder:
    def __init__(self, path, fps=30):
        self.writer = imageio.get_writer(path, fps=(1/config.FIXED_TIMESTEP))

    def capture(self, screen):
        arr = pygame.surfarray.array3d(screen)
        frame = np.transpose(arr, (1, 0, 2))  # Convert (W, H, C) -> (H, W, C)
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


class ScoreEffect:
    """Visual effect system for when someone scores"""
    def __init__(self):
        self.active = False
        self.effect_timer = 0.0
        self.effect_duration = 1.0  # 1 second effect
        self.flash_timer = 0.0
        self.particles = []
        self.scoring_side = None
        
    def trigger_score_effect(self, scoring_side):
        """Trigger the visual effect when someone scores"""
        self.active = True
        self.effect_timer = 0.0
        self.flash_timer = 0.0
        self.scoring_side = scoring_side
        self.particles = []
        
        # Create particles for the effect
        if scoring_side == 'left':
            # Particles from right wall (where ball hit)
            wall_x = config.SCREEN_WIDTH
            color = config.BLUE
        else:
            # Particles from left wall (where ball hit)
            wall_x = 0
            color = config.RED
            
        # Create explosion particles
        for _ in range(20):
            particle = {
                'x': wall_x,
                'y': random.randint(100, config.SCREEN_HEIGHT - 100),
                'vx': random.uniform(-200, 200) if scoring_side == 'left' else random.uniform(-200, 200),
                'vy': random.uniform(-150, 150),
                'life': 1.0,
                'color': color,
                'size': random.randint(3, 8)
            }
            self.particles.append(particle)
    
    def update(self, dt):
        """Update the visual effect"""
        if not self.active:
            return
            
        self.effect_timer += dt
        self.flash_timer += dt
        
        # Update particles
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['life'] -= dt / self.effect_duration
            particle['vy'] += 300 * dt  # Gravity effect
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
        
        # End effect when timer expires
        if self.effect_timer >= self.effect_duration:
            self.active = False
            self.particles = []
    
    def draw(self, screen):
        """Draw the visual effect"""
        if not self.active:
            return
            
        # Screen flash effect
        flash_intensity = max(0, 1.0 - (self.flash_timer / 0.3))  # Flash for 0.3 seconds
        if flash_intensity > 0:
            flash_alpha = int(flash_intensity * 100)
            flash_surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            if self.scoring_side == 'left':
                flash_surface.fill((0, 100, 255))  # Blue flash for left player score
            else:
                flash_surface.fill((255, 100, 0))  # Orange flash for right player score
            flash_surface.set_alpha(flash_alpha)
            screen.blit(flash_surface, (0, 0))
        
        # Draw particles
        for particle in self.particles:
            if particle['life'] > 0:
                alpha = int(particle['life'] * 255)
                size = int(particle['size'] * particle['life'])
                if size > 0:
                    # Create a surface for the particle with alpha
                    particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    color_with_alpha = (*particle['color'], alpha)
                    pygame.draw.circle(particle_surface, color_with_alpha, (size, size), size)
                    screen.blit(particle_surface, (int(particle['x'] - size), int(particle['y'] - size)))
        
        # Draw score text effect
        font = pygame.font.Font(None, 72)
        text_alpha = max(0, 1.0 - (self.effect_timer / 0.8))  # Text fades over 0.8 seconds
        if text_alpha > 0:
            if self.scoring_side == 'left':
                score_text = font.render("PLAYER 1 SCORES!", True, config.BLUE)
            else:
                score_text = font.render("PLAYER 2 SCORES!", True, config.RED)
            
            # Apply alpha to text
            text_surface = pygame.Surface(score_text.get_size(), pygame.SRCALPHA)
            text_surface.blit(score_text, (0, 0))
            text_surface.set_alpha(int(text_alpha * 255))
            
            text_rect = text_surface.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 - 50))
            screen.blit(text_surface, text_rect)


def create_episode_summary_table(episode, step_count, hits_p1, hits_p2, swings_p1, swings_p2, 
                                total_reward_p1, total_reward_p2, total_wins_p1, total_wins_p2, 
                                cumulative_total_avg_reward, episode_result=None):
    """
    Create and return an episode summary table.
    
    Args:
        episode_result: Optional string to indicate special episode endings (e.g., "TIMEOUT")
    """
    total_hits = hits_p1 + hits_p2
    total_swings = swings_p1 + swings_p2

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("", style="cyan")
    summary_table.add_column("", style="white")

    # Add episode result row if specified
    if episode_result:
        if episode_result == "TIMEOUT":
            summary_table.add_row("⏰ Episode Result", "[red]TIMEOUT - No Winner[/red]")
        else:
            summary_table.add_row("🏆 Episode Result", episode_result)
    summary_table.add_row("🏓 Step Count", f"{step_count}")
    summary_table.add_row("🏓 Hits", f"{total_hits}")

    summary_table.add_row("🥊 Swing %", f"{total_swings / (step_count * 2) * 100:.3f}%")
    summary_table.add_row("🏓 Hits", f"{total_hits}")
    summary_table.add_row("🏓 Swings", f"{total_swings}")
    summary_table.add_row("📈 P1 Reward", f"{total_reward_p1:.3f}")
    summary_table.add_row(
        "📈 P2 Reward", 
        f"{total_reward_p2:.3f}"
    )
    summary_table.add_row("🏆 Total Wins", f"P1: {total_wins_p1} | P2: {total_wins_p2}")

    return summary_table


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
    
    # Create score effect system
    score_effect = ScoreEffect()

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
            recorder = Recorder(f"videos/game_{episode:04}.mp4")

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
        
        # Score effect and pause timing
        post_score_pause_timer = 0.0
        is_post_score_pause = False
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
                # Draw white border around the outer perimeter
                border_width = 2
                pygame.draw.rect(screen, config.WHITE, (0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT), border_width)
                pygame.display.flip()
                clock.tick(10)
                continue
            
            # Handle post-score pause
            if is_post_score_pause:
                post_score_pause_timer += dt
                if post_score_pause_timer >= 0.5:  # 0.5 second pause
                    is_post_score_pause = False
                    post_score_pause_timer = 0.0
                
                # Continue rendering during pause but skip game logic
                if should_render:
                    # Update and render score effects during pause
                    score_effect.update(dt)
                    
                    screen.fill(config.BLACK)
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
                    
                    # Draw score effects
                    score_effect.draw(screen)
                    
                    # Draw white border
                    border_width = 2
                    pygame.draw.rect(screen, config.WHITE, (0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT), border_width)
                    
                    pygame.display.flip()
                    clock.tick(config.FPS)
                    if recorder:
                        recorder.capture(screen)
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
                # Trigger visual effects if rendering
                if should_render:
                    score_effect.trigger_score_effect(scorer)
                
                if scorer == 'left':
                    score1 += 1
                    reward1 += config.REWARD_WIN
                    reward2 += config.REWARD_LOSE
                else:
                    score2 += 1
                    reward1 += config.REWARD_LOSE
                    reward2 += config.REWARD_WIN
                
                ball.reset(scored_left=(scorer == 'right'))
                
                # Start post-score pause
                is_post_score_pause = True
                post_score_pause_timer = 0.0

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
                # Update score effects
                score_effect.update(dt)
                
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
                if vertical_ratio > config.BALL_VERTICAL_RATIO_THRESHOLD:
                    ratio_text = small_font.render(f"Vertical Ratio: {vertical_ratio:.2f} (threshold: {config.BALL_VERTICAL_RATIO_THRESHOLD})", True, ratio_color)
                    screen.blit(ratio_text, (10, info_y))
                    info_y += 22

                # Show anti-stall status (same logic as in ball.py increase_speed method)
                is_too_vertical = (abs_vx < config.BALL_MIN_X_VELOCITY) or \
                                 (abs_vx > 0 and abs_vy / abs_vx > config.BALL_VERTICAL_RATIO_THRESHOLD)
                stall_status = "ACTIVE" if is_too_vertical else "INACTIVE"
                stall_color = config.GREEN if is_too_vertical else config.GRAY
                if is_too_vertical:
                    stall_text = small_font.render(f"Anti-stall: {stall_status} (min_vx: {config.BALL_MIN_X_VELOCITY})", True, stall_color)
                    screen.blit(stall_text, (10, info_y))
                    info_y += 22

                eps_text = small_font.render(f"Epsilon: {agent1.epsilon:.3f}", True, config.WHITE)
                screen.blit(eps_text, (10, info_y))
                info_y += 20

                wins_text = small_font.render(f"Wins: P1={total_wins_p1}, P2={total_wins_p2}", True, config.WHITE)
                screen.blit(wins_text, (10, info_y))

                # Draw score effects (particles, flashes, text)
                score_effect.draw(screen)

                # Draw white border around the outer perimeter
                border_width = 2
                pygame.draw.rect(screen, config.WHITE, (0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT), border_width)

                pygame.display.flip()
                clock.tick(config.FPS)  # 60 FPS
                if recorder:
                    recorder.capture(screen)
        # Check if episode should end due to timeout
        if step_count >= config.MAX_STEPS_PER_EPISODE:
            episode_over = True

        # Handle episode ending (either by win/loss or timeout)
        if episode_over:
            # Handle timeout scenario
            if step_count >= config.MAX_STEPS_PER_EPISODE and score1 < config.WINNING_SCORE and score2 < config.WINNING_SCORE:
                # Both players get penalty for timeout
                timeout_reward1 = config.REWARD_GAME_LOSE
                timeout_reward2 = config.REWARD_GAME_LOSE

                # Accumulate timeout penalties
                total_reward_p1 += timeout_reward1
                total_reward_p2 += timeout_reward2

                # Create final experience for timeout
                final_next_state = np.array([
                    ball.rect.centerx / config.SCREEN_WIDTH,
                    ball.rect.centery / config.SCREEN_HEIGHT,
                    np.clip(ball.vx / config.BALL_MAX_SPEED, -1, 1),
                    np.clip(ball.vy / config.BALL_MAX_SPEED, -1, 1),
                    paddle1.rect.centery / config.SCREEN_HEIGHT,
                    paddle2.rect.centery / config.SCREEN_HEIGHT,
                    paddle1.swing_timer / config.PADDLE_SWING_DURATION if paddle1.swing_timer > 0 else 0,
                    paddle2.swing_timer / config.PADDLE_SWING_DURATION if paddle2.swing_timer > 0 else 0
                ], dtype=np.float32)

                # Store timeout experiences
                agent1.remember(state, action1, timeout_reward1, final_next_state, True)  # done=True for timeout
                agent2.remember(state, action2, timeout_reward2, final_next_state, True)  # done=True for timeout

                # Learn from timeout experience
                agent1.learn()
                agent2.learn()

                episode_result = "TIMEOUT"
                if should_render:
                    console.print(f"[bold red]⏰ Episode {episode} timed out after {step_count} steps - both players penalized[/bold red]")
            else:
                episode_result = None

            # Calculate cumulative average reward for episode summary
            avg_reward_p1 = total_reward_p1 / step_count if step_count > 0 else 0
            avg_reward_p2 = total_reward_p2 / step_count if step_count > 0 else 0
            total_avg_reward = avg_reward_p1 + avg_reward_p2
            cumulative_total_avg_reward = (cumulative_total_avg_reward * (episode - 1) + total_avg_reward) / episode
            if should_render:
                if recorder:
                    recorder.close()
            # Create and print episode summary table for all episode endings
            summary_table = create_episode_summary_table(
                episode, step_count, hits_p1, hits_p2, swings_p1, swings_p2,
                total_reward_p1, total_reward_p2, total_wins_p1, total_wins_p2,
                cumulative_total_avg_reward, episode_result
            )
            console.print(summary_table)

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

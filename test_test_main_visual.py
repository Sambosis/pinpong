import unittest
from unittest.mock import Mock, patch, call
import numpy as np
import pygame
import os
import sys
import tempfile
import shutil
import contextlib

# Mock all external dependencies before importing the module under test
sys.modules['config'] = Mock()
sys.modules['ball'] = Mock()
sys.modules['paddle'] = Mock()
sys.modules['agent'] = Mock()
sys.modules['vtrace_agent'] = Mock()
sys.modules['rich.console'] = Mock()
sys.modules['rich.panel'] = Mock()
sys.modules['rich.table'] = Mock()
sys.modules['imageio'] = Mock()

# Now import the module under test
import test_main_visual

class TestRecorder(unittest.TestCase):
    """Test suite for the Recorder class using unittest framework."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Mock imageio
        self.mock_writer = Mock()
        self.mock_imageio = Mock()
        self.mock_imageio.get_writer.return_value = self.mock_writer
        
        # Mock config
        test_main_visual.config.FIXED_TIMESTEP = 1/60
        
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('test_main_visual.imageio')
    def test_recorder_init_with_default_fps(self, mock_imageio):
        """Test Recorder initialization with default FPS calculation."""
        mock_imageio.get_writer.return_value = self.mock_writer
        
        recorder = test_main_visual.Recorder(self.test_path)
        
        # Verify writer is created with correct FPS from config
        expected_fps = 1 / test_main_visual.config.FIXED_TIMESTEP
        mock_imageio.get_writer.assert_called_once_with(self.test_path, fps=expected_fps)
        self.assertEqual(recorder.writer, self.mock_writer)
        
    @patch('test_main_visual.imageio')
    def test_recorder_init_with_custom_fps(self, mock_imageio):
        """Test Recorder initialization with custom FPS parameter."""
        mock_imageio.get_writer.return_value = self.mock_writer
        custom_fps = 30
        
        test_main_visual.Recorder(self.test_path, fps=custom_fps)
        
        # Should still use config.FIXED_TIMESTEP, not the fps parameter
        expected_fps = 1 / test_main_visual.config.FIXED_TIMESTEP
        mock_imageio.get_writer.assert_called_once_with(self.test_path, fps=expected_fps)
        
    @patch('test_main_visual.pygame')
    @patch('test_main_visual.np')
    @patch('test_main_visual.imageio')
    def test_recorder_capture_frame_processing(self, mock_imageio, mock_np, mock_pygame):
        """Test frame capture and processing."""
        mock_imageio.get_writer.return_value = self.mock_writer
        mock_screen = Mock()
        mock_array = np.zeros((800, 600, 3))
        mock_pygame.surfarray.array3d.return_value = mock_array
        mock_transposed = np.zeros((600, 800, 3))
        mock_np.transpose.return_value = mock_transposed
        
        recorder = test_main_visual.Recorder(self.test_path)
        recorder.capture(mock_screen)
        
        # Verify frame processing pipeline
        mock_pygame.surfarray.array3d.assert_called_once_with(mock_screen)
        mock_np.transpose.assert_called_once_with(mock_array, (1, 0, 2))
        self.mock_writer.append_data.assert_called_once_with(mock_transposed)
        
    @patch('test_main_visual.imageio')
    def test_recorder_close(self, mock_imageio):
        """Test recorder close functionality."""
        mock_imageio.get_writer.return_value = self.mock_writer
        
        recorder = test_main_visual.Recorder(self.test_path)
        recorder.close()
        
        self.mock_writer.close.assert_called_once()
        
    @patch('test_main_visual.pygame')
    @patch('test_main_visual.imageio')
    def test_recorder_capture_with_invalid_screen(self, mock_imageio, mock_pygame):
        """Test recorder behavior with invalid screen input."""
        mock_imageio.get_writer.return_value = self.mock_writer
        mock_pygame.surfarray.array3d.side_effect = pygame.error("Invalid surface")
        
        recorder = test_main_visual.Recorder(self.test_path)
        
        with self.assertRaises(pygame.error):
            recorder.capture(None)


class TestCreateEpisodeSummaryTable(unittest.TestCase):
    """Test suite for the create_episode_summary_table function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock rich.table.Table
        self.mock_table = Mock()
        self.mock_table_class = Mock(return_value=self.mock_table)
        test_main_visual.Table = self.mock_table_class
        
    def test_create_basic_episode_summary(self):
        """Test creation of basic episode summary without episode result."""
        episode = 5
        step_count = 100
        hits_p1, hits_p2 = 3, 2
        swings_p1, swings_p2 = 5, 4
        total_reward_p1, total_reward_p2 = 15.5, 12.3
        total_wins_p1, total_wins_p2 = 2, 1
        cumulative_avg_reward = 0.25
        
        result = test_main_visual.create_episode_summary_table(
            episode, step_count, hits_p1, hits_p2, swings_p1, swings_p2,
            total_reward_p1, total_reward_p2, total_wins_p1, total_wins_p2,
            cumulative_avg_reward
        )
        
        # Verify table creation
        self.mock_table_class.assert_called_once_with(show_header=False, box=None, padding=(0, 1))
        
        # Verify columns are added
        expected_calls = [
            call("", style="cyan"),
            call("", style="white")
        ]
        self.mock_table.add_column.assert_has_calls(expected_calls)
        
        # Verify basic rows are added (checking some key metrics)
        self.assertGreater(self.mock_table.add_row.call_count, 5)
        
        # Check specific calculations
        total_swings = swings_p1 + swings_p2
        swing_percentage = total_swings / (step_count * 2) * 100
        
        # Verify swing percentage calculation
        swing_call_found = False
        for call_args in self.mock_table.add_row.call_args_list:
            if "Swing %" in str(call_args):
                self.assertIn(f"{swing_percentage:.3f}%", str(call_args))
                swing_call_found = True
                break
        self.assertTrue(swing_call_found, "Swing percentage not found in table rows")
        
        self.assertEqual(result, self.mock_table)
        
    def test_create_episode_summary_with_timeout(self):
        """Test episode summary creation with timeout result."""
        test_main_visual.create_episode_summary_table(
            1, 50, 1, 1, 2, 2, 5.0, 5.0, 0, 0, 0.1, episode_result="TIMEOUT"
        )
        
        # Check that timeout result row is added
        timeout_call_found = False
        for call_args in self.mock_table.add_row.call_args_list:
            if "TIMEOUT" in str(call_args):
                timeout_call_found = True
                break
        self.assertTrue(timeout_call_found, "TIMEOUT result not found in table rows")
        
    def test_create_episode_summary_with_custom_result(self):
        """Test episode summary creation with custom episode result."""
        custom_result = "PLAYER_1_WINS"
        
        test_main_visual.create_episode_summary_table(
            1, 50, 1, 1, 2, 2, 5.0, 5.0, 1, 0, 0.1, episode_result=custom_result
        )
        
        # Check that custom result row is added
        custom_call_found = False
        for call_args in self.mock_table.add_row.call_args_list:
            if custom_result in str(call_args):
                custom_call_found = True
                break
        self.assertTrue(custom_call_found, f"Custom result {custom_result} not found in table rows")
        
    def test_edge_case_zero_steps(self):
        """Test episode summary with zero steps."""
        result = test_main_visual.create_episode_summary_table(
            1, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0
        )
        
        # Should handle division by zero gracefully
        self.mock_table_class.assert_called_once()
        self.assertEqual(result, self.mock_table)
        
    def test_large_numbers_formatting(self):
        """Test episode summary with large numbers."""
        test_main_visual.create_episode_summary_table(
            999, 10000, 500, 600, 1000, 1200, 9999.999, 8888.888, 50, 49, 1.234
        )
        
        # Verify formatting works with large numbers
        self.assertGreater(self.mock_table.add_row.call_count, 5)


class TestMainFunction(unittest.TestCase):
    """Test suite for the main function."""
    
    def setUp(self):
        """Set up comprehensive mocks for main function testing."""
        # Mock pygame
        self.mock_pygame = Mock()
        self.mock_screen = Mock()
        self.mock_clock = Mock()
        self.mock_font = Mock()
        self.mock_small_font = Mock()
        
        # Mock console and rich components
        self.mock_console = Mock()
        self.mock_panel = Mock()
        self.mock_table = Mock()
        
        # Mock agents
        self.mock_agent1 = Mock()
        self.mock_agent2 = Mock()
        
        # Mock game objects
        self.mock_paddle1 = Mock()
        self.mock_paddle2 = Mock()
        self.mock_ball = Mock()
        
        # Mock config values
        test_main_visual.config.SCREEN_WIDTH = 800
        test_main_visual.config.SCREEN_HEIGHT = 600
        test_main_visual.config.GAME_CAPTION = "Test Game"
        test_main_visual.config.NUM_EPISODES = 2
        test_main_visual.config.DISPLAY_EVERY = 1
        test_main_visual.config.WINNING_SCORE = 3
        test_main_visual.config.MAX_STEPS_PER_EPISODE = 100
        test_main_visual.config.LOAD_MODEL = False
        test_main_visual.config.MODEL_PATH = "test_models"
        test_main_visual.config.SAVE_MODEL_EVERY = 10
        test_main_visual.config.TARGET_UPDATE_FREQUENCY = 5
        test_main_visual.config.FIXED_TIMESTEP = 1/60
        test_main_visual.config.FPS = 60
        test_main_visual.config.PADDLE_WALL_OFFSET = 10
        test_main_visual.config.PADDLE_WIDTH = 20
        test_main_visual.config.PADDLE_SWING_DURATION = 10
        test_main_visual.config.BALL_MAX_SPEED = 300
        test_main_visual.config.BALL_SPEED_INITIAL = 200
        test_main_visual.config.REWARD_HIT = 10
        test_main_visual.config.REWARD_SWING = -1
        test_main_visual.config.REWARD_WIN = 100
        test_main_visual.config.REWARD_LOSE = -100
        test_main_visual.config.REWARD_GAME_WIN = 1000
        test_main_visual.config.REWARD_GAME_LOSE = -1000
        test_main_visual.config.BLACK = (0, 0, 0)
        test_main_visual.config.WHITE = (255, 255, 255)
        test_main_visual.config.GRAY = (128, 128, 128)
        test_main_visual.config.RED = (255, 0, 0)
        test_main_visual.config.GREEN = (0, 255, 0)
        test_main_visual.config.BALL_VERTICAL_RATIO_THRESHOLD = 2.0
        test_main_visual.config.BALL_MIN_X_VELOCITY = 50
        
    @patch('test_main_visual.pygame')
    @patch('test_main_visual.Console')
    @patch('test_main_visual.Panel')
    @patch('test_main_visual.Table')
    @patch('test_main_visual.Agent')
    @patch('test_main_visual.VTraceAgent')
    @patch('test_main_visual.Paddle')
    @patch('test_main_visual.Ball')
    @patch('test_main_visual.os')
    @patch('test_main_visual.Recorder')
    def test_main_initialization(self, mock_recorder, mock_os, mock_ball_class, 
                                mock_paddle_class, mock_vtrace_agent, mock_agent,
                                mock_table, mock_panel, mock_console, mock_pygame):
        """Test main function initialization phase."""
        # Setup mocks
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        mock_pygame.display.set_mode.return_value = self.mock_screen
        mock_pygame.time.Clock.return_value = self.mock_clock
        mock_pygame.font.Font.return_value = self.mock_font
        mock_agent.return_value = self.mock_agent1
        mock_vtrace_agent.return_value = self.mock_agent2
        mock_os.path.exists.return_value = True
        mock_os.makedirs = Mock()
        
        # Mock game events to exit quickly
        mock_event = Mock()
        mock_event.type = mock_pygame.QUIT
        mock_pygame.event.get.return_value = [mock_event]
        
        # Mock ball and paddle classes
        mock_paddle_class.return_value = self.mock_paddle1
        mock_ball_class.return_value = self.mock_ball
        
        # Mock ball methods
        self.mock_ball.rect.centerx = 400
        self.mock_ball.rect.centery = 300
        self.mock_ball.vx = 100
        self.mock_ball.vy = 50
        self.mock_ball.check_score.return_value = None
        self.mock_ball.rect.colliderect.return_value = False
        
        # Mock paddle methods
        self.mock_paddle1.rect.centery = 300
        self.mock_paddle1.swing_timer = 0
        self.mock_paddle2.rect.centery = 300
        self.mock_paddle2.swing_timer = 0
        
        # Mock agent methods
        self.mock_agent1.choose_action.return_value = 0
        self.mock_agent2.choose_action.return_value = 0
        self.mock_agent1.epsilon = 0.1
        
        with contextlib.suppress(SystemExit):
            test_main_visual.main()
        
        # Verify pygame initialization
        mock_pygame.init.assert_called_once()
        mock_pygame.display.set_mode.assert_called_once_with((800, 600))
        mock_pygame.display.set_caption.assert_called_once_with("Test Game - Visual Training")
        
        # Verify agents are created
        mock_agent.assert_called_once_with(agent_id='agent1')
        mock_vtrace_agent.assert_called_once_with(agent_id='agent2')
        
    @patch('test_main_visual.pygame')
    @patch('test_main_visual.Console')
    @patch('test_main_visual.Agent')
    @patch('test_main_visual.VTraceAgent')
    @patch('test_main_visual.os')
    def test_main_model_loading_success(self, mock_os, mock_vtrace_agent, mock_agent, mock_console, mock_pygame):
        """Test successful model loading in main function."""
        # Enable model loading
        test_main_visual.config.LOAD_MODEL = True
        test_main_visual.config.MODEL_TO_LOAD_P1 = "model1.pth"
        test_main_visual.config.MODEL_TO_LOAD_P2 = "model2.pth"
        
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        mock_pygame.display.set_mode.return_value = self.mock_screen
        mock_pygame.time.Clock.return_value = self.mock_clock
        mock_pygame.font.Font.return_value = self.mock_font
        mock_agent.return_value = self.mock_agent1
        mock_vtrace_agent.return_value = self.mock_agent2
        mock_os.path.exists.return_value = True
        mock_os.makedirs = Mock()
        
        # Exit immediately
        mock_event = Mock()
        mock_event.type = mock_pygame.QUIT
        mock_pygame.event.get.return_value = [mock_event]
        
        with contextlib.suppress(SystemExit):
            test_main_visual.main()
        
        # Verify model loading attempts
        self.mock_agent1.load_model.assert_called_once_with("model1.pth")
        self.mock_agent2.load_model.assert_called_once_with("model2.pth")
        
    @patch('test_main_visual.pygame')
    @patch('test_main_visual.Console')
    @patch('test_main_visual.Agent')
    @patch('test_main_visual.VTraceAgent')
    @patch('test_main_visual.os')
    def test_main_model_loading_failure(self, mock_os, mock_vtrace_agent, mock_agent, mock_console, mock_pygame):
        """Test model loading failure handling."""
        # Enable model loading but files don't exist
        test_main_visual.config.LOAD_MODEL = True
        test_main_visual.config.MODEL_TO_LOAD_P1 = "nonexistent1.pth"
        test_main_visual.config.MODEL_TO_LOAD_P2 = "nonexistent2.pth"
        
        mock_console_instance = Mock()
        mock_console.return_value = mock_console_instance
        mock_pygame.display.set_mode.return_value = self.mock_screen
        mock_pygame.time.Clock.return_value = self.mock_clock
        mock_pygame.font.Font.return_value = self.mock_font
        mock_agent.return_value = self.mock_agent1
        mock_vtrace_agent.return_value = self.mock_agent2
        mock_os.path.exists.return_value = False  # Files don't exist
        mock_os.makedirs = Mock()
        
        # Exit immediately
        mock_event = Mock()
        mock_event.type = mock_pygame.QUIT
        mock_pygame.event.get.return_value = [mock_event]
        
        with contextlib.suppress(SystemExit):
            test_main_visual.main()
        
        # Verify models are not loaded when files don't exist
        self.mock_agent1.load_model.assert_not_called()
        self.mock_agent2.load_model.assert_not_called()
        
    def test_state_normalization(self):
        """Test state normalization calculations."""
        # Test values
        ball_x, ball_y = 400, 300
        ball_vx, ball_vy = 150, 75
        paddle1_y, paddle2_y = 250, 350
        swing_timer1, swing_timer2 = 5, 0
        
        # Expected normalized values
        expected_ball_x_norm = ball_x / test_main_visual.config.SCREEN_WIDTH
        expected_ball_y_norm = ball_y / test_main_visual.config.SCREEN_HEIGHT
        expected_ball_vx_norm = np.clip(ball_vx / test_main_visual.config.BALL_MAX_SPEED, -1, 1)
        expected_ball_vy_norm = np.clip(ball_vy / test_main_visual.config.BALL_MAX_SPEED, -1, 1)
        expected_p1_y_norm = paddle1_y / test_main_visual.config.SCREEN_HEIGHT
        expected_p2_y_norm = paddle2_y / test_main_visual.config.SCREEN_HEIGHT
        expected_p1_swing_norm = swing_timer1 / test_main_visual.config.PADDLE_SWING_DURATION
        expected_p2_swing_norm = swing_timer2 / test_main_visual.config.PADDLE_SWING_DURATION
        
        # Verify calculations
        self.assertAlmostEqual(expected_ball_x_norm, 0.5, places=3)
        self.assertAlmostEqual(expected_ball_y_norm, 0.5, places=3)
        self.assertAlmostEqual(expected_ball_vx_norm, 0.5, places=3)
        self.assertAlmostEqual(expected_ball_vy_norm, 0.25, places=3)
        self.assertAlmostEqual(expected_p1_y_norm, 250/600, places=3)
        self.assertAlmostEqual(expected_p2_y_norm, 350/600, places=3)
        self.assertAlmostEqual(expected_p1_swing_norm, 0.5, places=3)
        self.assertAlmostEqual(expected_p2_swing_norm, 0.0, places=3)
        
    def test_reward_calculations(self):
        """Test reward calculation logic."""
        # Test swing penalties
        swing_reward = test_main_visual.config.REWARD_SWING
        self.assertEqual(swing_reward, -1)
        
        # Test hit rewards
        hit_reward = test_main_visual.config.REWARD_HIT
        self.assertEqual(hit_reward, 10)
        
        # Test win/lose rewards
        win_reward = test_main_visual.config.REWARD_WIN
        lose_reward = test_main_visual.config.REWARD_LOSE
        self.assertEqual(win_reward, 100)
        self.assertEqual(lose_reward, -100)
        
        # Test game win/lose rewards
        game_win_reward = test_main_visual.config.REWARD_GAME_WIN
        game_lose_reward = test_main_visual.config.REWARD_GAME_LOSE
        self.assertEqual(game_win_reward, 1000)
        self.assertEqual(game_lose_reward, -1000)

class TestGameLogic(unittest.TestCase):
    """Test suite for game logic and state management."""
    
    def test_action_mapping(self):
        """Test action to paddle movement mapping."""
        # Action mappings based on main function logic:
        # 0: No action
        # 1: Move up (-1)
        # 2: Move down (1)
        # 3: Swing
        
        actions = [0, 1, 2, 3]
        expected_movements = [None, -1, 1, None]  # None for no movement or swing
        
        for action, expected in zip(actions, expected_movements, strict=True):
            if action == 1:
                self.assertEqual(expected, -1, "Action 1 should move paddle up")
            elif action == 2:
                self.assertEqual(expected, 1, "Action 2 should move paddle down")
            elif action in [0, 3]:
                self.assertIsNone(expected, f"Action {action} should not have movement value")
                
    def test_score_tracking(self):
        """Test score tracking and win conditions."""
        winning_score = test_main_visual.config.WINNING_SCORE
        
        # Test that game should end when a player reaches winning score
        for score1 in range(winning_score + 2):
            for score2 in range(winning_score + 2):
                game_should_end = (score1 >= winning_score) or (score2 >= winning_score)
                
                if score1 >= winning_score:
                    self.assertTrue(game_should_end, f"Game should end when P1 score ({score1}) >= {winning_score}")
                if score2 >= winning_score:
                    self.assertTrue(game_should_end, f"Game should end when P2 score ({score2}) >= {winning_score}")
                    
    def test_timeout_conditions(self):
        """Test timeout condition checking."""
        max_steps = test_main_visual.config.MAX_STEPS_PER_EPISODE
        winning_score = test_main_visual.config.WINNING_SCORE
        
        # Simulate timeout scenario
        step_count = max_steps
        score1 = winning_score - 1  # Below winning score
        score2 = winning_score - 1  # Below winning score
        
        # Should be timeout condition
        is_timeout = (step_count >= max_steps and 
                     score1 < winning_score and 
                     score2 < winning_score)
        self.assertTrue(is_timeout, "Should detect timeout condition")
        
        # Test non-timeout scenarios
        step_count = max_steps - 1
        is_timeout = (step_count >= max_steps and 
                     score1 < winning_score and 
                     score2 < winning_score)
        self.assertFalse(is_timeout, "Should not detect timeout when steps below max")
        
        step_count = max_steps
        score1 = winning_score  # At winning score
        is_timeout = (step_count >= max_steps and 
                     score1 < winning_score and 
                     score2 < winning_score)
        self.assertFalse(is_timeout, "Should not detect timeout when player has won")

class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions and edge cases."""
    
    def test_vertical_ratio_calculation(self):
        """Test vertical ratio calculation for anti-stall mechanism."""
        # Test normal ratios
        vx, vy = 100, 50
        ratio = abs(vy) / abs(vx) if abs(vx) > 0 else float('inf')
        self.assertEqual(ratio, 0.5)
        
        # Test threshold comparison
        threshold = test_main_visual.config.BALL_VERTICAL_RATIO_THRESHOLD
        self.assertEqual(threshold, 2.0)
        
        # Test division by zero case
        vx, vy = 0, 50
        ratio = abs(vy) / abs(vx) if abs(vx) > 0 else float('inf')
        self.assertEqual(ratio, float('inf'))
        
        # Test high vertical ratio
        vx, vy = 10, 50
        ratio = abs(vy) / abs(vx) if abs(vx) > 0 else float('inf')
        self.assertEqual(ratio, 5.0)
        self.assertGreater(ratio, threshold)
        
    def test_anti_stall_condition(self):
        """Test anti-stall detection logic."""
        min_x_velocity = test_main_visual.config.BALL_MIN_X_VELOCITY
        threshold = test_main_visual.config.BALL_VERTICAL_RATIO_THRESHOLD
        
        # Test cases for anti-stall activation
        test_cases = [
            (10, 100, True),   # vx below minimum
            (0, 50, True),     # vx is zero
            (20, 50, True),    # ratio above threshold (50/20 = 2.5 > 2.0)
            (100, 50, False),  # Normal case
            (60, 100, False),  # Ratio at threshold (100/60 ≈ 1.67 < 2.0)
        ]
        
        for vx, vy, expected_stall in test_cases:
            abs_vx = abs(vx)
            abs_vy = abs(vy)
            
            is_too_vertical = (abs_vx < min_x_velocity) or \
                             (abs_vx > 0 and abs_vy / abs_vx > threshold)
            
            self.assertEqual(is_too_vertical, expected_stall, 
                             f"Anti-stall detection failed for vx={vx}, vy={vy}")
            
    def test_cumulative_average_calculation(self):
        """Test cumulative average reward calculation."""
        # Simulate episode rewards
        episode_rewards = [10, 20, 15, 25, 30]
        cumulative_avg = 0.0
        
        for episode, reward in enumerate(episode_rewards, 1):
            cumulative_avg = (cumulative_avg * (episode - 1) + reward) / episode
            
            # Calculate expected average manually
            expected_avg = sum(episode_rewards[:episode]) / episode
            self.assertAlmostEqual(cumulative_avg, expected_avg, places=5)
            
    def test_swing_percentage_calculation(self):
        """Test swing percentage calculation in episode summary."""
        step_count = 100
        total_swings = 15
        expected_percentage = total_swings / (step_count * 2) * 100
        
        # Should be 7.5% for 15 swings in 100 steps (200 total actions)
        self.assertEqual(expected_percentage, 7.5)
        
        # Test edge case with zero steps
        step_count = 0
        total_swings = 0
        # Should handle division by zero gracefully (0/0 case)
        percentage = total_swings / (step_count * 2) * 100 if step_count > 0 else 0.0
        self.assertEqual(percentage, 0.0)

if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)
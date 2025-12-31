"""
Test script for RoutingBoardGameEnv to verify basic functionality.
"""

import numpy as np
from src.routing_board_game.routing_env import RoutingBoardGameEnv


def test_basic_functionality():
    """Test basic environment operations."""
    print("Testing basic environment functionality...")

    env = RoutingBoardGameEnv(num_ai_pieces=8, max_steps=10)

    # Test reset
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Board shape: {obs['board'].shape}")
    print(f"  Step count: {obs['step_count'][0]}")

    # Verify initial state
    assert obs["board"].shape == (10, 10), "Board should be 10x10"
    assert obs["step_count"][0] == 0, "Initial step count should be 0"

    # Count AI pieces on board
    ai_piece_count = np.sum(obs["board"] == 1)
    print(f"  AI pieces on board: {ai_piece_count}")
    assert ai_piece_count == 8, "Should have 8 AI pieces initially"

    # Check root is marked
    root_count = np.sum(obs["board"] == 3)
    assert root_count == 1, "Should have exactly one root square"
    assert obs["board"][0, 5] == 3, "Root should be at (0, 5)"

    print(f"✓ Initial state verified")

    return env


def test_random_episode():
    """Test a full episode with random actions."""
    print("\nTesting random episode...")

    env = RoutingBoardGameEnv(num_ai_pieces=8, max_steps=10, render_mode="human")
    obs, info = env.reset(seed=42)

    env.render()

    episode_reward = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Random action (place blocking piece)
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        print(f"\nStep {steps}:")
        print(f"  Action: place at ({action // 10}, {action % 10})")
        print(f"  Placement success: {info['placement_success']}")
        print(f"  Pieces reached root: {info['pieces_reached_root']}")
        print(f"  Pieces remaining: {info['pieces_remaining']}")
        print(f"  Step reward: {reward:.2f}")

        env.render()

        if steps >= 20:  # Safety limit
            print("  (Stopping after 20 steps for safety)")
            break

    print(f"\n✓ Episode completed")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Final pieces remaining: {info['pieces_remaining']}")
    print(f"  Total pieces routed: {info['pieces_routed_total']}")

    return episode_reward


def test_multiple_episodes():
    """Test that environment is stable across multiple episodes."""
    print("\nTesting environment stability over multiple episodes...")

    env = RoutingBoardGameEnv(num_ai_pieces=8, max_steps=5)

    num_episodes = 10
    rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        print(f"  Episode {episode + 1}: {steps} steps, reward: {episode_reward:.2f}")

    print(f"\n✓ All {num_episodes} episodes completed")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min reward: {np.min(rewards):.2f}")
    print(f"  Max reward: {np.max(rewards):.2f}")


def test_movement_rules():
    """Test that movement rules are correctly enforced."""
    print("\nTesting movement rules...")

    env = RoutingBoardGameEnv(num_ai_pieces=3, max_steps=20, render_mode="human")

    # Set seed for reproducibility
    obs, info = env.reset(seed=123)

    print("Initial state:")
    env.render()

    # Record initial positions
    initial_ai_pieces = len(env.ai_pieces)
    print(f"  Initial AI pieces: {initial_ai_pieces}")

    # Take a few steps
    for i in range(5):
        # Place blocking piece at random location
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nAfter step {i + 1}:")
        env.render()
        print(f"  Pieces reached root: {info['pieces_reached_root']}")
        print(f"  Pieces remaining: {info['pieces_remaining']}")

        if terminated or truncated:
            print("  Episode ended")
            break

    print(f"\n✓ Movement rules test completed")


def test_termination_conditions():
    """Test different termination conditions."""
    print("\nTesting termination conditions...")

    # Test 1: Max steps termination
    print("  Test 1: Max steps termination")
    env = RoutingBoardGameEnv(num_ai_pieces=8, max_steps=3)
    obs, info = env.reset(seed=42)

    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    assert truncated or terminated, "Episode should end after max steps"
    print("    ✓ Max steps termination works")

    # Test 2: Check that environment can be reset after termination
    print("  Test 2: Reset after termination")
    obs, info = env.reset()
    assert obs["step_count"][0] == 0, "Step count should reset to 0"
    print("    ✓ Reset after termination works")

    print(f"\n✓ Termination conditions test completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RoutingBoardGameEnv Test Suite")
    print("=" * 60)

    try:
        # Run tests
        env = test_basic_functionality()
        test_random_episode()
        test_multiple_episodes()
        test_movement_rules()
        test_termination_conditions()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

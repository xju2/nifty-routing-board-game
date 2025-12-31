#!/usr/bin/env python3
"""
Example script demonstrating the RoutingBoardGameEnv.

This script shows:
1. How to create and reset the environment
2. How to take actions (place blocking pieces)
3. How the AI routing phase works
4. How to interpret observations and rewards
"""

from routing_board_game.routing_env import RoutingBoardGameEnv


def main():
    print("=" * 70)
    print("Routing Board Game Environment - Example Usage")
    print("=" * 70)
    print()

    # Create environment
    print("Creating environment...")
    env = RoutingBoardGameEnv(
        num_ai_pieces=5,  # Use fewer pieces for easier visualization
        max_steps=10,
        reward_shaping=True,
        render_mode="human",
    )
    print("✓ Environment created\n")

    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset(seed=42)
    print("✓ Environment reset\n")

    # Show initial state
    print("Initial State:")
    print("-" * 70)
    env.render()

    # Run a few steps
    episode_reward = 0
    step = 0

    print("\nRunning episode with random actions...\n")

    while True:
        step += 1

        # Sample random action (place blocking piece at random position)
        action = env.action_space.sample()
        row, col = action // 10, action % 10

        print(f"Step {step}:")
        print(f"  Action: Place blocking piece at ({row}, {col})")

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Show results
        print(f"  Placement success: {info['placement_success']}")
        print(f"  Pieces reached root this turn: {info['pieces_reached_root']}")
        print(f"  Pieces remaining: {info['pieces_remaining']}")
        print(f"  Reward this step: {reward:.2f}")
        print()

        # Render new state
        env.render()

        # Check if episode ended
        if terminated:
            print("\n" + "=" * 70)
            print("EPISODE TERMINATED")
            if info["pieces_remaining"] == 0:
                print("Result: SUCCESS - All AI pieces reached the root!")
            else:
                print("Result: FAILURE - AI pieces are blocked!")
            print("=" * 70)
            break

        if truncated:
            print("\n" + "=" * 70)
            print("EPISODE TRUNCATED - Max steps reached")
            print("=" * 70)
            break

    # Print final statistics
    print("\nFinal Statistics:")
    print("-" * 70)
    print(f"  Total steps: {step}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  AI pieces routed to root: {info['pieces_routed_total']}")
    print(f"  AI pieces remaining: {info['pieces_remaining']}")
    print(f"  Blocking pieces placed: {len(env.blocking_pieces)}")
    print()

    # Demonstrate observation space
    print("Observation Space:")
    print("-" * 70)
    print(f"  Board shape: {obs['board'].shape}")
    print(f"  Board encoding:")
    print(f"    0 = empty")
    print(f"    1 = AI piece")
    print(f"    2 = blocking piece")
    print(f"    3 = root")
    print(f"  Step count: {obs['step_count'][0]}")
    print()

    # Demonstrate action space
    print("Action Space:")
    print("-" * 70)
    print(f"  Type: Discrete(100)")
    print(f"  Actions represent grid positions:")
    print(f"    Action n -> position (n // 10, n % 10)")
    print(f"  Example: Action 45 -> position (4, 5)")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

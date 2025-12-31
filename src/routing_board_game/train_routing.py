"""
Training script for the RoutingBoardGameEnv using Stable-Baselines3.

This script trains a reinforcement learning agent to place blocking pieces
to prevent AI pieces from reaching the root.
"""

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from routing_board_game.routing_env import RoutingBoardGameEnv
import os


def train_routing_agent(
    num_ai_pieces: int = 8,
    max_steps: int = 10,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    algorithm: str = "PPO",
    log_dir: str = "./logs/",
    tensorboard_log: str = "./tensorboard_logs/"
):
    """
    Train an RL agent for the routing board game.
    
    Args:
        num_ai_pieces: Number of AI pieces to route (default: 8)
        max_steps: Maximum steps per episode (default: 10)
        total_timesteps: Total training timesteps (default: 100,000)
        n_envs: Number of parallel environments (default: 4)
        algorithm: RL algorithm to use - "PPO" or "DQN" (default: "PPO")
        log_dir: Directory for saving models and logs
        tensorboard_log: Directory for tensorboard logs
    """
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  AI Pieces: {num_ai_pieces}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Total Timesteps: {total_timesteps}")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Algorithm: {algorithm}")
    print()
    
    # Create vectorized environment
    def make_env():
        return RoutingBoardGameEnv(
            num_ai_pieces=num_ai_pieces,
            max_steps=max_steps,
            reward_shaping=True
        )
    
    env = make_vec_env(make_env, n_envs=n_envs)
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = RoutingBoardGameEnv(
        num_ai_pieces=num_ai_pieces,
        max_steps=max_steps,
        reward_shaping=True
    )
    
    # Select and configure algorithm
    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log,
        )
    elif algorithm.upper() == "DQN":
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=1000,
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'PPO' or 'DQN'")
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(total_timesteps // 20, 1000),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 1000),
        save_path=log_dir,
        name_prefix=f"routing_{algorithm.lower()}_model"
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False
    )
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(log_dir, f"final_{algorithm.lower()}_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Run a demo episode
    print("\n" + "=" * 60)
    print("Running demonstration episode with trained model...")
    print("=" * 60)
    
    demo_env = RoutingBoardGameEnv(
        num_ai_pieces=num_ai_pieces,
        max_steps=max_steps,
        render_mode="human"
    )
    
    obs, info = demo_env.reset(seed=42)
    demo_env.render()
    
    total_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"\nStep {step_count}:")
        print(f"  Action: place blocking at ({action // 10}, {action % 10})")
        print(f"  Reward: {reward:.2f}")
        print(f"  Info: {info}")
        
        demo_env.render()
    
    print("\n" + "=" * 60)
    print(f"Demo Episode Results:")
    print(f"  Total Steps: {step_count}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Pieces Routed: {info['pieces_routed_total']}")
    print(f"  Pieces Remaining: {info['pieces_remaining']}")
    print(f"  Success: {info['pieces_remaining'] == 0}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train routing board game agent")
    parser.add_argument("--num_ai_pieces", type=int, default=8,
                        help="Number of AI pieces (default: 8)")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Maximum steps per episode (default: 10)")
    parser.add_argument("--total_timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100,000)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "DQN"],
                        help="RL algorithm to use (default: PPO)")
    parser.add_argument("--log_dir", type=str, default="./logs/",
                        help="Directory for logs and models (default: ./logs/)")
    
    args = parser.parse_args()
    
    train_routing_agent(
        num_ai_pieces=args.num_ai_pieces,
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        algorithm=args.algorithm,
        log_dir=args.log_dir
    )

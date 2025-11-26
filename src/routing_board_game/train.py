from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from routing_board_game.game_env import RoutingGameEnv


def train(placer_extra_pieces):
    # Create the environment
    # We wrap it in a Vectorized Environment for faster training
    env = make_vec_env(
        lambda: RoutingGameEnv(placer_extra_pieces=placer_extra_pieces), n_envs=4
    )

    # Instantiate the agent
    # Using MultiInputPolicy because our observation is a Dict
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log="./routing_board_tensorboard/",
    )

    # Define a callback to evaluate performance periodically
    eval_env = RoutingGameEnv(placer_extra_pieces=placer_extra_pieces)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    print("Starting training...")
    # Train the agent
    model.learn(total_timesteps=100_000, callback=eval_callback)
    print("Training finished.")

    model.save("ppo_router_agent")

    # --- Demonstration of Trained Agent ---
    print("\nRunning demonstration...")
    obs = eval_env.reset()[0]
    terminated = False
    total_reward = 0

    while not terminated:
        eval_env.render()
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

    print(f"Game Over. Final Reward (Negative Score): {total_reward}")

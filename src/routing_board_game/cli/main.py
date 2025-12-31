import click
from routing_board_game.ai_server import start_route_ai_server
from routing_board_game.train import train as _train
from routing_board_game.play_game import play_game as _play_game
from routing_board_game.train_routing import train_routing_agent


@click.group()
def main():
    """Routing Board Game CLI."""


# add training command for old routing game
@click.command("train")
@click.option(
    "--placer_extra_pieces",
    default=5,
    help="Number of extra pieces the placer can add.",
)
@click.option(
    "--total_timesteps",
    default=100_000,
    help="Total timesteps for training the agent.",
)
def train(placer_extra_pieces, total_timesteps):
    """Train the routing board game agent (old arrow-based game)."""
    _train(placer_extra_pieces, total_timesteps)


# add training command for new RL environment
@click.command("train-rl")
@click.option(
    "--num_ai_pieces",
    default=8,
    help="Number of AI pieces to route.",
)
@click.option(
    "--max_steps",
    default=10,
    help="Maximum steps per episode.",
)
@click.option(
    "--total_timesteps",
    default=100_000,
    help="Total timesteps for training the agent.",
)
@click.option(
    "--n_envs",
    default=4,
    help="Number of parallel environments.",
)
@click.option(
    "--algorithm",
    default="PPO",
    type=click.Choice(["PPO", "DQN"], case_sensitive=False),
    help="RL algorithm to use.",
)
@click.option(
    "--log_dir",
    default="./logs/",
    help="Directory for logs and models.",
)
def train_rl(num_ai_pieces, max_steps, total_timesteps, n_envs, algorithm, log_dir):
    """Train the routing board game agent (new multi-move RL environment)."""
    train_routing_agent(
        num_ai_pieces=num_ai_pieces,
        max_steps=max_steps,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        algorithm=algorithm,
        log_dir=log_dir
    )


# add play command
@click.command("play")
@click.option(
    "--model_path",
    default="ppo_router_agent_single_file.zip",
    help="Path to the trained model file.",
)
def play(model_path):
    """Play the routing board game against the trained agent."""
    _play_game(model_path)


# start AI routing server
@click.command("server")
@click.option(
    "--model_path",
    default="ppo_router_agent_single_file.zip",
    show_default=True,
    help="Path to the trained model file.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host interface to bind the AI server.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind the AI server.",
)
@click.option(
    "--base_path",
    default="",
    show_default=True,
    help="Optional base path if served behind a path prefix (e.g., /nifty-ai).",
)
def start_route_ai(model_path, host, port, base_path):
    """Start a Flask server that serves routing actions."""
    try:
        start_route_ai_server(
            model_path=model_path, host=host, port=port, base_path=base_path
        )
    except Exception as exc:
        raise click.ClickException(str(exc))


main.add_command(train)
main.add_command(train_rl)
main.add_command(play)
main.add_command(start_route_ai)


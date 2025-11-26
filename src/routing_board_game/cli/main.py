import click
from routing_board_game.train import train as _train
from routing_board_game.play_game import play_game as _play_game


@click.group()
def main():
    """Routing Board Game CLI."""


# add training command
@click.command("train")
@click.option(
    "--placer_extra_pieces",
    default=5,
    help="Number of extra pieces the placer can add.",
)
def train(placer_extra_pieces):
    """Train the routing board game agent."""
    _train(placer_extra_pieces)


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


main.add_command(train)
main.add_command(play)

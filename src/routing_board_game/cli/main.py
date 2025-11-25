import click
from routing_board_game.train import train as _train


@click.group()
def main():
    """Routing Board Game CLI."""


# add training command
@click.command("train")
@click.option('--placer_extra_pieces', default=5, help='Number of extra pieces the placer can add.')
def train(placer_extra_pieces):
    """Train the routing board game agent."""
    _train(placer_extra_pieces)

main.add_command(train)
"""
Updated tests for RoutingBoardGameEnv with new rules.
"""

import numpy as np
from routing_board_game.routing_env import RoutingBoardGameEnv
from routing_board_game.policies import BaseRoutingPolicy


def test_initial_state():
    env = RoutingBoardGameEnv()
    obs, _ = env.reset(seed=0)
    assert env.user_pieces_remaining == 100
    assert len(env.ai_pieces) == 0
    assert np.sum(obs["board"] == 1) == 0
    assert obs["board"][env.root_pos] == 3


def test_place_until_empty():
    env = RoutingBoardGameEnv()
    obs, _ = env.reset(seed=1)
    # Place a single piece and step once
    action = 11  # (1,1)
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["placement_success"]
    assert env.user_pieces_remaining == 99
    # Eventually terminates when pieces leave the board
    for _ in range(200):
        if terminated:
            break
        obs, reward, terminated, truncated, info = env.step(action)
    assert terminated


def test_collision_penalty():
    env = RoutingBoardGameEnv()
    env.reset(seed=2)
    # Manually set pieces to collide on next move
    env.ai_pieces = [(2, 5), (2, 6)]
    env.board = np.zeros((10, 10), dtype=np.uint8)
    env.board[env.root_pos] = 3
    env.board[2, 5] = 1
    env.board[2, 6] = 1

    # Force both to move to (1,5) so one is eaten
    class CollisionPolicy(BaseRoutingPolicy):
        def plan_moves(self, *args, **kwargs):
            return [((2, 5), (1, 5)), ((2, 6), (1, 5))]

    env.move_policy = CollisionPolicy()
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["eaten_pieces"] >= 1
    assert reward < -1


def test_success_bonus():
    env = RoutingBoardGameEnv(success_bonus=25.0)
    env.reset(seed=3)
    # Place one piece next to root so it routes immediately
    env.ai_pieces = [(1, 5)]
    env.board = np.zeros((10, 10), dtype=np.uint8)
    env.board[env.root_pos] = 3
    env.board[1, 5] = 1
    # Use an invalid placement on root to avoid adding new pieces
    root_action = env.root_pos[0] * env.grid_size + env.root_pos[1]
    obs, reward, terminated, truncated, info = env.step(root_action)
    assert terminated
    assert reward >= env.success_bonus - 1  # includes per-turn cost


if __name__ == "__main__":
    test_initial_state()
    test_place_until_empty()
    test_collision_penalty()
    test_success_bonus()
    print("Tests completed")

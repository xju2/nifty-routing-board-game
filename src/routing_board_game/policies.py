"""Routing policies for deciding how to move all AI pieces in a turn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np

Position = Tuple[int, int]
ForwardMoveFn = Callable[[Position, np.ndarray], List[Position]]


class BaseRoutingPolicy(ABC):
    """Abstract policy controlling the routing phase (all pieces in a turn)."""

    @abstractmethod
    def plan_moves(
        self,
        ai_pieces: List[Position],
        board: np.ndarray,
        root_pos: Position,
        forward_move_fn: ForwardMoveFn,
        rng: np.random.Generator,
    ) -> List[Tuple[Position, Position]]:
        """
        Return an ordered list of (src, dst) moves for this routing phase.
        Each piece should move at most once; dst must be in forward_move_fn(src).

        Why `forward_move_fn`? Policies often simulate moves on a board copy to
        account for blocking/unblocking as pieces move. They must query the same
        legality function the environment uses so every proposed destination is
        valid under current occupancy and the forward-move rules.
        """


class GreedyRoutingPolicy(BaseRoutingPolicy):
    """
    Default greedy hill-climbing policy (former built-in behavior).

    Iteratively picks any piece with a forward move; if multiple moves exist,
    chooses one uniformly at random. Recomputes blocking after every move.
    """

    def plan_moves(
        self,
        ai_pieces: List[Position],
        board: np.ndarray,
        root_pos: Position,
        forward_move_fn: ForwardMoveFn,
        rng: np.random.Generator,
    ) -> List[Tuple[Position, Position]]:
        moves: List[Tuple[Position, Position]] = []
        board_state = board.copy()
        pieces_to_move = list(ai_pieces)

        while pieces_to_move:
            moved_this_iteration = False
            for i, pos in enumerate(pieces_to_move):
                forward_moves = forward_move_fn(pos, board_state)
                if not forward_moves:
                    continue

                if len(forward_moves) == 1:
                    dst = forward_moves[0]
                else:
                    dst = forward_moves[rng.integers(0, len(forward_moves))]

                # apply move on board_state to keep occupancy updated
                if pos != root_pos:
                    board_state[pos] = 0
                if dst != root_pos:
                    board_state[dst] = 1

                moves.append((pos, dst))
                pieces_to_move.pop(i)
                moved_this_iteration = True
                break

            if not moved_this_iteration:
                break  # remaining pieces cannot move

        return moves


class NeuralNetRoutingPolicy(BaseRoutingPolicy):
    """
    Example NN-backed policy: scores forward moves for each piece.

    Provide `model_fn(board, piece_pos, forward_moves) -> scores`. The highest
    scoring move is taken for each piece in order. Blocking is recomputed after
    every move on the policy's working board copy.
    """

    def __init__(self, model_fn):
        self.model_fn = model_fn

    def plan_moves(
        self,
        ai_pieces: List[Position],
        board: np.ndarray,
        root_pos: Position,
        forward_move_fn: ForwardMoveFn,
        rng: np.random.Generator,
    ) -> List[Tuple[Position, Position]]:
        moves: List[Tuple[Position, Position]] = []
        board_state = board.copy()
        pieces_to_move = list(ai_pieces)

        while pieces_to_move:
            moved_this_iteration = False
            for i, pos in enumerate(pieces_to_move):
                forward_moves = forward_move_fn(pos, board_state)
                if not forward_moves:
                    continue

                try:
                    scores = self.model_fn(board_state, pos, forward_moves)
                    best_idx = int(np.argmax(list(scores)))
                    dst = forward_moves[best_idx]
                except Exception:
                    dst = forward_moves[rng.integers(0, len(forward_moves))]

                if pos != root_pos:
                    board_state[pos] = 0
                if dst != root_pos:
                    board_state[dst] = 1

                moves.append((pos, dst))
                pieces_to_move.pop(i)
                moved_this_iteration = True
                break

            if not moved_this_iteration:
                break

        return moves

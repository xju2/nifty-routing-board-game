"""Move selection policies for AI pieces during the routing phase.

Policies encapsulate how an AI piece chooses among legal forward moves. The
environment keeps turn/episode logic; a policy only decides which move to take
for a single piece when given the set of forward moves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

Position = Tuple[int, int]


class BaseMovePolicy(ABC):
    """Abstract base for AI move selection."""

    @abstractmethod
    def select_move(
        self,
        piece_pos: Position,
        forward_moves: List[Position],
        board: np.ndarray,
        root_pos: Position,
        rng: np.random.Generator,
    ) -> Position:
        """
        Pick the destination for a piece given its legal forward moves.
        Must return one of the provided `forward_moves`.
        """


class GreedyPolicy(BaseMovePolicy):
    """Default policy: pick a random forward move (greedy hill-climb)."""

    def select_move(
        self,
        piece_pos: Position,
        forward_moves: List[Position],
        board: np.ndarray,
        root_pos: Position,
        rng: np.random.Generator,
    ) -> Position:
        if not forward_moves:
            raise ValueError("No forward moves available for selection")
        if len(forward_moves) == 1:
            return forward_moves[0]
        return forward_moves[rng.integers(0, len(forward_moves))]


class NeuralNetPolicy(BaseMovePolicy):
    """
    Example NN-backed policy wrapper.

    Pass a callable `model_fn(board, piece_pos, forward_moves)` that returns
    a score for each forward move (aligned with the list order). The move with
    the highest score is selected.
    """

    def __init__(self, model_fn):
        self.model_fn = model_fn

    def select_move(
        self,
        piece_pos: Position,
        forward_moves: List[Position],
        board: np.ndarray,
        root_pos: Position,
        rng: np.random.Generator,
    ) -> Position:
        if not forward_moves:
            raise ValueError("No forward moves available for selection")
        scores = self.model_fn(board, piece_pos, forward_moves)
        # Expect iterable of scores; fall back to greedy if malformed
        try:
            best_idx = int(np.argmax(list(scores)))
            return forward_moves[best_idx]
        except Exception:
            return forward_moves[rng.integers(0, len(forward_moves))]

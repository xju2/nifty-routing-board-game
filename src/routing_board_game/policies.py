"""Routing policies for deciding how to move all AI pieces in a turn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Move every piece once if possible
        for pos in list(ai_pieces):
            forward_moves = forward_move_fn(pos, board_state)
            if not forward_moves:
                continue
            if len(forward_moves) == 1:
                dst = forward_moves[0]
            else:
                dst = forward_moves[rng.integers(0, len(forward_moves))]
            if pos != root_pos:
                board_state[pos] = 0
            if dst != root_pos:
                board_state[dst] = 1
            moves.append((pos, dst))
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


class TorchRoutingPolicy(BaseRoutingPolicy):
    """
    Torch-backed routing policy that also exposes log-probs for policy gradient.

    Expects a torch model with signature model(board_flat, src, dst) -> score.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.last_log_probs: List[torch.Tensor] = []

    def _score_moves(
        self, board_state: np.ndarray, src: Position, forward_moves: List[Position]
    ) -> torch.Tensor:
        board_flat = torch.tensor(
            board_state.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        src_tensor = torch.tensor(
            [[src[0], src[1]]], dtype=torch.float32, device=self.device
        )
        dst_tensor = torch.tensor(
            [[m[0], m[1]] for m in forward_moves],
            dtype=torch.float32,
            device=self.device,
        )
        board_batch = board_flat.repeat(dst_tensor.shape[0], 1)
        src_batch = src_tensor.repeat(dst_tensor.shape[0], 1)
        scores = self.model(board_batch, src_batch, dst_tensor).squeeze(-1)
        return scores

    def plan_moves(
        self,
        ai_pieces: List[Position],
        board: np.ndarray,
        root_pos: Position,
        forward_move_fn: ForwardMoveFn,
        rng: np.random.Generator,
    ) -> List[Tuple[Position, Position]]:
        self.last_log_probs = []
        moves: List[Tuple[Position, Position]] = []
        board_state = board.copy()
        pieces_to_move = list(ai_pieces)

        while pieces_to_move:
            moved_this_iteration = False
            for i, pos in enumerate(pieces_to_move):
                forward_moves = forward_move_fn(pos, board_state)
                if not forward_moves:
                    continue

                with torch.no_grad():
                    scores = self._score_moves(board_state, pos, forward_moves)
                probs = F.softmax(scores, dim=0)
                dist = torch.distributions.Categorical(probs)
                idx = int(dist.sample().item())
                dst = forward_moves[idx]
                self.last_log_probs.append(
                    dist.log_prob(torch.tensor(idx, device=probs.device))
                )

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

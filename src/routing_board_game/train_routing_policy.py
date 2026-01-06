"""Train a routing policy (NeuralNetRoutingPolicy) using REINFORCE.

This trains a small MLP to score candidate forward moves for each piece. The
policy is plugged into the environment via `move_policy` so it controls the
entire routing phase.

Usage:
    python -m routing_board_game.train_routing_policy --episodes 2000
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from routing_board_game.routing_env import RoutingBoardGameEnv
from routing_board_game.policies import TorchRoutingPolicy

Position = Tuple[int, int]


class MLPScoreModel(nn.Module):
    """Simple MLP that scores (src, dst, board) triples."""

    def __init__(self, board_size: int = 10, hidden: int = 128):
        super().__init__()
        input_dim = board_size * board_size + 4  # board flattened + src(rc) + dst(rc)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, board_flat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        # board_flat: (B, board_size*board_size)
        x = torch.cat([board_flat, src, dst], dim=-1)
        return self.net(x).squeeze(-1)


def episode(
    env: RoutingBoardGameEnv, policy: TorchRoutingPolicy, action_noop: int
) -> Tuple[float, List[torch.Tensor]]:
    log_probs: List[torch.Tensor] = []
    obs, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        env.move_policy = policy
        # take a no-op placement (on root) to avoid adding pieces
        obs, reward, terminated, truncated, info = env.step(action_noop)
        total_reward += reward
        log_probs.extend(policy.last_log_probs)
    return total_reward, log_probs


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    env = RoutingBoardGameEnv(
        num_ai_pieces=args.num_ai_pieces, max_steps=args.max_steps
    )
    model = MLPScoreModel(board_size=env.grid_size, hidden=args.hidden).to(device)
    policy = TorchRoutingPolicy(model, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    action_noop = env.root_pos[0] * env.grid_size + env.root_pos[1]

    for episode_idx in range(1, args.episodes + 1):
        total_reward, log_probs = episode(env, policy, action_noop)

        if log_probs:
            loss = -total_reward * torch.stack(log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode_idx % args.log_interval == 0:
            print(
                f"Episode {episode_idx}: reward={total_reward:.2f}, moves={len(log_probs)}"
            )

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved trained routing model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralNet routing policy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num_ai_pieces", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--save_path", type=str, default="routing_policy.pt")
    args = parser.parse_args()
    train(args)

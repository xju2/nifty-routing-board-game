#!/usr/bin/env python3
"""
Pygame-based GUI to visualize the routing board game.

Controls:
  - Left click: place a user piece on the clicked cell
  - Space: place a random user piece
  - R: reset the environment
  - Q or ESC: quit

After each action the AI moves its pieces sequentially. The GUI highlights each
piece as it moves (source red, destination green) so you can see the per-piece
updates within a turn.
"""

import sys
import pygame
import math
import numpy as np

from routing_board_game.routing_env import RoutingBoardGameEnv
from routing_board_game.routing_policy_env import RoutingPolicyEnv


CELL_SIZE = 60
MARGIN = 2
SIDEBAR_WIDTH = 240
FPS = 60
MOVE_DELAY_MS = 420
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 40

# Colors
BG_COLOR = (20, 24, 28)
GRID_COLOR = (60, 70, 80)
AI_COLOR = (66, 179, 245)
ROOT_COLOR = (255, 204, 0)
TEXT_COLOR = (225, 230, 235)
HIGHLIGHT_SRC = (232, 80, 91)
HIGHLIGHT_DST = (80, 200, 120)


def draw_board(
    screen,
    font,
    board,
    root_pos,
    step_info,
    current_turn,
    highlight=None,
    bounce_positions=None,
):
    """Draw the grid, pieces, and sidebar info."""
    screen.fill(BG_COLOR)
    grid_size = board.shape[0]
    bounce_positions = bounce_positions or set()
    t_ms = pygame.time.get_ticks()

    # Draw cells
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * CELL_SIZE + MARGIN
            y = row * CELL_SIZE + MARGIN
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            pos = (row, col)
            # Root
            if pos == root_pos:
                pygame.draw.rect(screen, ROOT_COLOR, rect)
            # AI piece
            if board[row, col] == 1:
                if pos in bounce_positions:
                    offset = int(4 * (1 + math.sin(t_ms / 180)))
                else:
                    offset = 0
                bounce_rect = rect.inflate(-10, -10).move(0, -offset)
                pygame.draw.ellipse(screen, AI_COLOR, bounce_rect)

    # Highlight current move
    if highlight:
        src, dst = highlight
        for pos, color in [(src, HIGHLIGHT_SRC), (dst, HIGHLIGHT_DST)]:
            x = pos[1] * CELL_SIZE + MARGIN
            y = pos[0] * CELL_SIZE + MARGIN
            rect = pygame.Rect(x, y, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)
            pygame.draw.rect(screen, color, rect, 3)

        # Draw a line from src to dst
        src_center = (
            src[1] * CELL_SIZE + CELL_SIZE // 2,
            src[0] * CELL_SIZE + CELL_SIZE // 2,
        )
        dst_center = (
            dst[1] * CELL_SIZE + CELL_SIZE // 2,
            dst[0] * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.line(screen, HIGHLIGHT_SRC, src_center, dst_center, 4)

    # Sidebar
    sidebar_x = grid_size * CELL_SIZE + 20
    lines = [
        f"Step: {step_info.get('step', 0)} / {step_info.get('max_steps', '?')}",
        f"Reward: {step_info.get('reward', 0):.2f}",
        f"Pieces remaining: {step_info.get('pieces_remaining', 0)}",
        f"Pieces routed: {step_info.get('pieces_routed_total', 0)}",
        f"Pieces spawned: {step_info.get('pieces_spawned_total', 0)}",
        f"User pieces added: {step_info.get('user_pieces_added', 0)}",
        f"User pieces remaining: {step_info.get('user_pieces_remaining', 0)}",
        f"Eaten pieces last turn: {step_info.get('eaten_pieces', 0)}",
        "",
        f"Placement success: {step_info.get('placement_success', False)}",
        f"Terminated: {step_info.get('terminated', False)}",
        f"Truncated: {step_info.get('truncated', False)}",
        "",
        f"Current turn: {current_turn}",
    ]

    for i, line in enumerate(lines):
        text = font.render(line, True, TEXT_COLOR)
        screen.blit(text, (sidebar_x, 20 + i * 24))

    # End Turn button (clickable)
    button_rect = pygame.Rect(
        grid_size * CELL_SIZE + 20,
        20 + len(lines) * 24 + 10,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
    )
    pygame.draw.rect(screen, (90, 140, 210), button_rect, border_radius=6)
    text = font.render(current_turn.capitalize() + " turn", True, BG_COLOR)
    text_rect = text.get_rect(center=button_rect.center)
    screen.blit(text, text_rect)

    pygame.display.flip()
    return button_rect


def draw_summary_overlay(screen, font, info, reward):
    """Draw an end-of-game summary overlay inside the Pygame window."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))

    summary_rect = pygame.Rect(0, 0, 420, 260)
    summary_rect.center = screen.get_rect().center
    pygame.draw.rect(overlay, (240, 240, 240), summary_rect, border_radius=10)
    pygame.draw.rect(overlay, (90, 140, 210), summary_rect, width=3, border_radius=10)

    lines = [
        "GAME OVER",
        f"Steps: {info.get('step_count', '?')}",
        f"Reward: {reward:.2f}",
        f"Pieces routed: {info.get('pieces_routed_total', 0)}",
        f"Pieces remaining: {info.get('pieces_remaining', 0)}",
        f"Pieces placed: {info.get('user_pieces_added', 0)}",
        f"User pieces remaining: {info.get('user_pieces_remaining', 0)}",
        f"Pieces eaten: {info.get('eaten_pieces', 0)}",
    ]

    for idx, line in enumerate(lines):
        color = BG_COLOR if idx == 0 else (30, 30, 30)
        text = font.render(line, True, color)
        text_rect = text.get_rect(
            center=(summary_rect.centerx, summary_rect.top + 30 + idx * 28)
        )
        overlay.blit(text, text_rect)

    screen.blit(overlay, (0, 0))
    pygame.display.flip()


def animate_moves(
    screen,
    font,
    board_before,
    root_pos,
    step_info,
    move_sequence,
    placement_pos=None,
):
    """Animate the sequence of AI moves for the current turn."""
    temp_board = board_before.copy()
    if placement_pos is not None and temp_board[placement_pos] == 0:
        # Show the newly placed piece before any AI moves occur
        temp_board[placement_pos] = 1

    draw_board(screen, font, temp_board, root_pos, step_info, current_turn="ai")
    pygame.time.delay(MOVE_DELAY_MS // 2)

    for src, dst in move_sequence:
        # Show highlight before move
        draw_board(
            screen,
            font,
            temp_board,
            root_pos,
            step_info,
            current_turn="ai",
            highlight=(src, dst),
        )
        pygame.time.delay(MOVE_DELAY_MS)

        # Apply move on the temp board for visualization
        if temp_board[src] == 1:
            temp_board[src] = 0
        if dst != root_pos:
            temp_board[dst] = 1

        draw_board(
            screen,
            font,
            temp_board,
            root_pos,
            step_info,
            current_turn="ai",
            highlight=(src, dst),
        )
        pygame.time.delay(MOVE_DELAY_MS)


def run_gui():
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)

    use_policy_env = "--policy" in sys.argv
    env = (
        RoutingPolicyEnv(render_mode=None)
        if use_policy_env
        else RoutingBoardGameEnv(render_mode=None)
    )
    obs, info = env.reset(seed=42)

    board = obs["board"]
    grid_size = env.grid_size

    screen = pygame.display.set_mode(
        (grid_size * CELL_SIZE + SIDEBAR_WIDTH, grid_size * CELL_SIZE)
    )
    pygame.display.set_caption("Routing Board Game - Pygame Viewer")

    last_reward = 0.0
    last_info = info
    last_terminated = False
    last_truncated = False
    current_turn = "user"  # user or ai
    pending_actions = []  # only used in policy env user turn

    running = True
    run_gui._pending_src = None
    last_moved_positions = set()
    step_info = {}
    button_rect = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check button click
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if "button_rect" in locals() and button_rect.collidepoint(
                    mouse_x, mouse_y
                ):
                    # if current_turn == "ai":
                    #     # run the pending AI moves.
                    #     if isinstance(env, RoutingPolicyEnv):
                    #         obs, reward, terminated, truncated, info = env.step(pending_actions)

                    current_turn = "ai" if current_turn == "user" else "user"
                    continue

                if last_terminated or last_truncated:
                    continue
                mouse_x, mouse_y = pygame.mouse.get_pos()
                col = mouse_x // CELL_SIZE
                row = mouse_y // CELL_SIZE

                if 0 <= row < grid_size and 0 <= col < grid_size:
                    if current_turn == "ai":
                        # two-click selection: first selects src, second selects dst
                        if getattr(run_gui, "_pending_src", None) is None:
                            run_gui._pending_src = (row, col)
                        else:
                            src = run_gui._pending_src
                            dst = (row, col)
                            action = (
                                src[0] * grid_size + src[1],
                                dst[0] * grid_size + dst[1],
                            )
                            animate_moves(
                                screen,
                                font,
                                board,
                                env.root_pos,
                                step_info,
                                [(src, dst)],
                            )
                            pending_actions.append(action)
                            run_gui._pending_src = None
                    else:
                        action = row * grid_size + col
                        obs, reward, terminated, truncated, info = env.step(action)
                        placement_pos = (
                            (row, col) if info["placement_success"] else None
                        )
                        step_info = {
                            "step": info["step_count"],
                            "max_steps": env.max_steps,
                            "reward": reward,
                            **info,
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                        animate_moves(
                            screen,
                            font,
                            board,
                            env.root_pos,
                            step_info,
                            info["move_sequence"],
                            placement_pos=placement_pos,
                        )
                        board = obs["board"]
                        last_reward = reward
                        last_info = info
                        last_terminated = terminated
                        last_truncated = truncated
                        last_moved_positions = {
                            dst for _, dst in info.get("move_sequence", [])
                        }

            # Update display
            current_info = {
                "step": last_info.get("step_count", 0),
                "max_steps": env.max_steps,
                "reward": last_reward,
                **last_info,
                "terminated": last_terminated,
                "truncated": last_truncated,
            }
            bounce_positions = set()
            if isinstance(env, RoutingPolicyEnv) and not (
                last_terminated or last_truncated
            ):
                # Bounce movable AI pieces during AI turn
                if current_turn == "ai":
                    bounce_positions = {
                        tuple(pos) for pos in zip(*np.where(board == 1))
                    }
                    bounce_positions.difference_update(last_moved_positions)
            button_rect = draw_board(
                screen,
                font,
                board,
                env.root_pos,
                current_info,
                current_turn=current_turn,
                bounce_positions=bounce_positions,
            )
            clock.tick(FPS)

        if last_terminated or last_truncated:
            draw_summary_overlay(screen, font, last_info, last_reward)
            # Stay open until user closes the window
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            continue

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_gui()

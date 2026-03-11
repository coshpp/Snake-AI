"""
play_ai.py — Watch the trained agent play without any training.

Run:
  python3 play_ai.py — default 20x20 grid
  python3 play_ai.py <size> — size x size grid 

Keybindings:
  Q / ESC — quit
  R — reset
"""

import sys
import numpy as np
import pygame

from snake_env import SnakeGame, DEFAULT_GRID_SIZE
from game import Renderer
from model import load_model, ONLINE_MODEL_PATH, ACTION_SIZE

SPEEDS = [10, 30, 60, 120, 1_000_000]

def play_ai(grid_size: int = DEFAULT_GRID_SIZE, speed: int = 10):
    model = load_model(ONLINE_MODEL_PATH)
    if model is None:
        print(f"No model found at {ONLINE_MODEL_PATH}. Train first with: python3 train.py")
        sys.exit(1)

    game = SnakeGame(grid_size=grid_size, human=False, speed=speed)
    renderer = Renderer(game)
    state = game.reset()
    speed_idx = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    state = game.reset()
                if event.key == pygame.K_UP:
                    speed_idx = (speed_idx + 1) % len(SPEEDS)
                    game.speed = SPEEDS[speed_idx]
                if event.key == pygame.K_DOWN:
                    speed_idx = (speed_idx - 1) % len(SPEEDS)
                    game.speed = SPEEDS[speed_idx]

        q_values = model(state[np.newaxis], training=False).numpy()[0]
        action = int(np.argmax(q_values))

        state, _, done = game.step(action)
        if done:
            state = game.reset()

        renderer.draw()


if __name__ == "__main__":
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
    play_ai(grid_size=grid_size)

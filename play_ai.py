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

from snake_env import SnakeGame, FPS_AI, DEFAULT_GRID_SIZE
from game import Renderer
from model import load_model, MODEL_PATH, ACTION_SIZE


def play_ai(grid_size: int = DEFAULT_GRID_SIZE, speed: int = FPS_AI):
    model = load_model(MODEL_PATH)
    if model is None:
        print(f"No model found at {MODEL_PATH}. Train first with: python train.py")
        sys.exit(1)

    game = SnakeGame(grid_size=grid_size, human=False, speed=speed)
    renderer = Renderer(game)
    state = game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    state = game.reset()

        q_values = model(state[np.newaxis], training=False).numpy()[0]
        action = int(np.argmax(q_values))

        state, _, done = game.step(action)
        if done:
            state = game.reset()

        renderer.draw()


if __name__ == "__main__":
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
    play_ai(grid_size=grid_size)
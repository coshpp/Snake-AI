"""
snake_env.py — Snake game logic and environment.

This is the game logic itself with no visuals
The agent interacts with it like this every step:
  1. get_state() — what does the snake currently see?
  2. step(action) — move the snake, get back a reward and whether it died
  3. reset() - start a new game
"""

import random
from enum import Enum
from collections import deque

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GRID_SIZE = 20
FPS_HUMAN = 10
FPS_AI = 1000000

# ── Direction ─────────────────────────────────────────────────────────────────

class Dir(Enum):
    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)

# Directions listed clockwise 
CW = [Dir.RIGHT, Dir.DOWN, Dir.LEFT, Dir.UP]

# ── Snake Game ────────────────────────────────────────────────────────────────

class SnakeGame:
    """
    The Snake environment.

    The agent sees 11 binary values as its state:
      [0-2]   danger ahead, right, left
      [3-6]   current direction (one-hot)
      [7-10]  food left, right, above, below
    """

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE, human: bool = True, speed: int = None):
        self.grid_size = grid_size
        self.human = human
        self.speed = speed or (FPS_HUMAN if human else FPS_AI)
        self.record = 0
        self.reset()

    def reset(self) -> np.ndarray:
        # Place the snake in the middle of the board, facing right
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.direction = Dir.RIGHT
        self.snake = deque([
            (cx, cy),
            (cx - 1, cy),
            (cx - 2, cy),
        ])
        self.score = 0
        self.steps = 0
        # End the game if the snake hasn't eaten in this many steps
        self.max_steps = self.grid_size * self.grid_size * 2
        self._place_food()
        return self.get_state()

    def step(self, action: int):
        """
        Move the snake one step.
        action: 0 = keep going straight, 1 = turn right, 2 = turn left
        Returns: (new state, reward, did the game end?)
        """
        self.steps += 1
        self._apply_action(action)

        head = self.snake[0]

        if self._is_collision(head):
            return self.get_state(), -10, True

        if head == self.food:
            self.steps = 0
            self.score += 1
            if self.score > self.record:
                self.record = self.score
            self._place_food()
            return self.get_state(), 10, False
        else:
            self.snake.pop()  

        # Took too long without eating — end the game to prevent looping
        if self.steps >= self.max_steps:
            return self.get_state(), -10, True

        return self.get_state(), 0, False

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        idx = CW.index(self.direction)

        straight = CW[idx]
        right_d = CW[(idx + 1) % 4]
        left_d = CW[(idx - 1) % 4]

        def ahead(d):
            return (head[0] + d.value[0], head[1] + d.value[1])

        state = [
            # Danger flags 
            int(self._is_collision(ahead(straight))),
            int(self._is_collision(ahead(right_d))),
            int(self._is_collision(ahead(left_d))),
            # Current direction 
            int(self.direction == Dir.LEFT),
            int(self.direction == Dir.RIGHT),
            int(self.direction == Dir.UP),
            int(self.direction == Dir.DOWN),
            # Food location 
            int(self.food[0] < head[0]),
            int(self.food[0] > head[0]),
            int(self.food[1] < head[1]),
            int(self.food[1] > head[1]),
        ]
        return np.array(state, dtype=np.float32)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        idx = CW.index(self.direction)
        if action == 1:
            self.direction = CW[(idx + 1) % 4]
        elif action == 2:
            self.direction = CW[(idx - 1) % 4]

        dx, dy = self.direction.value
        hx, hy = self.snake[0]
        self.snake.appendleft((hx + dx, hy + dy))

    def _is_collision(self, pos) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        return pos in list(self.snake)[1:]

    def _place_food(self):
        snake_set = set(self.snake)
        while True:
            pos = (random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1))
            if pos not in snake_set:
                self.food = pos
                break

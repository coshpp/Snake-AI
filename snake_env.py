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

    The agent sees 20 values as its state:
    [0-15]  ray signals (8 relative rays × 2 values: wall distance, body distance)
            rays are relative to heading: forward, forward-right, right, backward-right,
            backward, backward-left, left, forward-left
    [16-17] food vector (food_forward, food_right) normalized by grid size
    [18-19] tail vector (tail_forward, tail_right) normalized by grid size
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
        self.steps = 0                # total steps this episode
        self.steps_since_food = 0     # starvation counter
        # End the game if the snake hasn't eaten in this many steps
        self.max_steps_without_food = self.grid_size * self.grid_size * 2
        self._place_food()
        head = self.snake[0]
        self.best_food_dist = abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])
        return self.get_state()

    def step(self, action: int):
        self.steps += 1
        self.steps_since_food += 1
        self._apply_action(action)

        head = self.snake[0]
        ate_food = (head == self.food)

        # If not eating, tail moves away this turn
        if not ate_food:
            self.snake.pop()

        if self._is_collision(head):
            return self.get_state(), -10, True

        if ate_food:
            self.steps_since_food = 0
            self.score += 1
            if self.score > self.record:
                self.record = self.score
            self._place_food()

            # reset progress tracker for the new food
            self.best_food_dist = abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])

            return self.get_state(), 10, False

        if self.steps_since_food >= self.max_steps_without_food:
            return self.get_state(), -10, True

        curr_dist = abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])

        reward = -0.01
        if curr_dist < self.best_food_dist:
            reward += 0.1
            self.best_food_dist = curr_dist

        return self.get_state(), reward, False

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        snake_body = set(list(self.snake)[1:])

        # Raycasting 
        ray_inputs = []
        for ray_dir in self._get_ray_dirs():
            wall_dist, body_dist = self._cast_ray(head, ray_dir, snake_body)
            ray_inputs.extend([wall_dist, body_dist])

        # Food position relative to head
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]

        fx, fy = self.direction.value      # forward direction
        rx, ry = -fy, fx                   # right direction (90° clockwise)

        food_forward = (dx * fx + dy * fy) / self.grid_size
        food_right = (dx * rx + dy * ry) / self.grid_size

        # Tail position relative to head
        tail = self.snake[-1]
        tdx = tail[0] - head[0]
        tdy = tail[1] - head[1]

        tail_forward = (tdx * fx + tdy * fy) / self.grid_size
        tail_right = (tdx * rx + tdy * ry) / self.grid_size

        state = ray_inputs + [food_forward, food_right, tail_forward, tail_right]
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

    def _cast_ray(self, origin, direction, snake_body):
        dx, dy = direction
        x, y = origin
        distance = 0
        body_signal = 0.0  # 0 means no body seen on this ray

        while True:
            distance += 1
            x += dx
            y += dy

            if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
                wall_signal = 1.0 / distance
                return wall_signal, body_signal

            if body_signal == 0.0 and (x, y) in snake_body:
                body_signal = 1.0 / distance

    def _get_ray_dirs(self):
        fx, fy = self.direction.value
        rx, ry = -fy, fx      # right (clockwise)
        lx, ly = fy, -fx      # left (counter-clockwise)
        bx, by = -fx, -fy     # backward

        return [
            (fx, fy),                 # forward
            (fx + rx, fy + ry),       # forward-right
            (rx, ry),                 # right
            (bx + rx, by + ry),       # backward-right
            (bx, by),                 # backward
            (bx + lx, by + ly),       # backward-left
            (lx, ly),                 # left
            (fx + lx, fy + ly),       # forward-left
        ]
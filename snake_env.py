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
FPS_AI = 1_000_000

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
    [20-22] available space for next move
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
        self.snake_set = set(self.snake)
        self.score = 0
        self.steps = 0                # total steps this episode
        self.steps_since_food = 0     # starvation counter
        # End the game if the snake hasn't eaten in this many steps
        self.max_steps_without_food = self.grid_size * self.grid_size * 2
        self._place_food()
        return self.get_state()

    def step(self, action: int):
        self.steps += 1
        self.steps_since_food += 1

        new_dir = self._resolve_direction(action)
        dx, dy = new_dir.value
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        ate_food = (new_head == self.food)

        if self._is_collision(new_head, ate_food):
            self.direction = new_dir
            return self.get_state(), -10, True

        self.direction = new_dir
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)

        if ate_food:
            self.steps_since_food = 0
            self.score += 1
            if self.score > self.record:
                self.record = self.score
            self._place_food()
            return self.get_state(), 10, False

        tail = self.snake.pop()
        self.snake_set.discard(tail)

        if self.steps_since_food >= self.max_steps_without_food:
            return self.get_state(), -10, True

        prev_dist = abs(self.food[0] - hx) + abs(self.food[1] - hy)
        curr_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])

        reward = -0.01
        if curr_dist < prev_dist:
            reward += 0.1
        elif curr_dist > prev_dist:
            reward -= 0.1

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

        # Reachable space — build both blocked sets once
        blocked_no_eat = set(list(self.snake)[:-1])  # tail moves, so exclude it
        blocked_eat = set(self.snake)                 # tail stays, so include it
        max_visit = len(self.snake) * 3
        spaces = []

        for action in [0, 1, 2]:
            next_head = self._next_head(action)
            ate_food = (next_head == self.food)
            blocked = blocked_eat if ate_food else blocked_no_eat

            if self._is_collision(next_head, ate_food):
                spaces.append(0.0)
            else:
                reachable = self._reachable_space(next_head, blocked, max_visit)
                spaces.append(reachable / max_visit)

        state = ray_inputs + [food_forward, food_right, tail_forward, tail_right] + spaces
        return np.array(state, dtype=np.float32)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_direction(self, action: int) -> Dir:
        idx = CW.index(self.direction)
        if action == 1:
            return CW[(idx + 1) % 4]
        elif action == 2:
            return CW[(idx - 1) % 4]
        return self.direction

    def _is_collision(self, pos, ate_food=False) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True

        old_tail = self.snake[-1]

        if pos in self.snake_set:
            return not (not ate_food and pos == old_tail)

        return False

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

    def _reachable_space(self, start, blocked, max_visit):
        if start in blocked:
            return 0

        queue = deque([start])
        visited = {start}

        while queue:
            if len(visited) >= max_visit:
                return max_visit

            x, y = queue.popleft()

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                pos = (nx, ny)

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                if pos in blocked or pos in visited:
                    continue

                visited.add(pos)
                queue.append(pos)

        return len(visited)
    
    def _next_head(self, action):
        new_dir = self._resolve_direction(action)
        dx, dy = new_dir.value
        hx, hy = self.snake[0]
        return (hx + dx, hy + dy)
"""
snake_env.py — Snake game logic and environment.

The agent interacts with the game like this every step:
  1. get_state() — observe what the snake currently sees (24 floats)
  2. step(action) — move the snake, get back (state, reward, done)
  3. reset()     — start a new game
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

    State (24 floats):
      [0-15]  8 rays × 2 signals each (wall distance, body distance)
      [16-17] food direction (forward, right) relative to heading
      [18-20] reachable space per action (straight, right, left)
      [21-23] tail reachable per action (1 = yes, 0 = no)

    Actions: 0 = straight, 1 = turn right, 2 = turn left

    Rewards:
      +10   eat food
      -10   die (collision or starvation)
      -0.01 + fading proximity shaping (off by length 50)
    """

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE, human: bool = True, speed: int = None):
        self.grid_size = grid_size
        self.speed = speed or (FPS_HUMAN if human else FPS_AI)
        self.record = 0
        self.reset()

    # ── Core interface ────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new game. Returns the initial state."""
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.direction = Dir.RIGHT
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.snake_set = set(self.snake)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.max_steps_without_food = self.grid_size * self.grid_size * 2
        self._place_food()
        return self.get_state()

    def step(self, action: int) -> tuple:
        """
        Move the snake one step.

        Args:
            action: 0 = straight, 1 = turn right, 2 = turn left

        Returns:
            (state, reward, done)
        """
        self.steps += 1
        self.steps_since_food += 1

        new_dir = self._resolve_direction(action)
        dx, dy = new_dir.value
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)
        ate_food = (new_head == self.food)

        # Death by collision
        if self._is_collision(new_head, ate_food):
            self.direction = new_dir
            return self.get_state(), -10, True

        self.direction = new_dir
        self.snake.appendleft(new_head)
        self.snake_set.add(new_head)

        # Ate food — grow
        if ate_food:
            self.steps_since_food = 0
            self.score += 1
            if self.score > self.record:
                self.record = self.score
            self._place_food()
            return self.get_state(), 10, False

        # Normal move — remove tail
        old_tail = self.snake.pop()
        if old_tail != new_head:
            self.snake_set.discard(old_tail)

        # Death by starvation
        if self.steps_since_food >= self.max_steps_without_food:
            return self.get_state(), -10, True

        # Reward: small step cost + fading proximity shaping
        reward = -0.01
        shaping_weight = max(0.0, 1.0 - len(self.snake) / 50.0)
        if shaping_weight > 0:
            prev_dist = abs(self.food[0] - hx) + abs(self.food[1] - hy)
            curr_dist = abs(self.food[0] - new_head[0]) + abs(self.food[1] - new_head[1])
            if curr_dist < prev_dist:
                reward += 0.1 * shaping_weight
            elif curr_dist > prev_dist:
                reward -= 0.1 * shaping_weight

        return self.get_state(), reward, False

    def get_state(self) -> np.ndarray:
        """Build the 24-float state vector the agent sees."""
        head = self.snake[0]
        tail = self.snake[-1]
        body = self.snake_set - {head}

        # ── Rays (16 values) ──
        ray_inputs = []
        for ray_dir in self._get_ray_dirs():
            wall_dist, body_dist = self._cast_ray(head, ray_dir, body)
            ray_inputs.extend([wall_dist, body_dist])

        # ── Food direction (2 values) ──
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]
        fx, fy = self.direction.value       # forward
        rx, ry = -fy, fx                    # right (90° clockwise)
        food_forward = (dx * fx + dy * fy) / self.grid_size
        food_right = (dx * rx + dy * ry) / self.grid_size

        # ── Per-action BFS: reachable space + tail reachability (6 values) ──
        blocked_no_eat = self.snake_set - {tail}    # normal move: tail will vacate
        blocked_eat = set(self.snake_set)            # eating: tail stays
        free_cells = self.grid_size * self.grid_size - len(self.snake)
        bfs_cap = min(free_cells, 150)
        skip_bfs = len(self.snake) <= 20

        spaces = []
        tail_reachable = []

        for a in (0, 1, 2):
            next_head = self._next_head(a)
            ate_food = (next_head == self.food)
            blocked = blocked_eat if ate_food else blocked_no_eat

            if self._is_collision(next_head, ate_food):
                spaces.append(0.0)
                tail_reachable.append(0.0)
            elif skip_bfs:
                spaces.append(1.0)
                tail_reachable.append(1.0)
            else:
                count, found_tail = self._flood_fill(next_head, blocked, bfs_cap, tail)
                spaces.append(count / max(1, free_cells))
                tail_reachable.append(1.0 if found_tail else 0.0)

        return np.array(
            ray_inputs + [food_forward, food_right] + spaces + tail_reachable,
            dtype=np.float32,
        )

    # ── Movement helpers ──────────────────────────────────────────────────────

    def _resolve_direction(self, action: int) -> Dir:
        """Convert a relative action (0/1/2) to an absolute direction."""
        idx = CW.index(self.direction)
        if action == 1:
            return CW[(idx + 1) % 4]  # right
        if action == 2:
            return CW[(idx - 1) % 4]  # left
        return self.direction          # straight

    def _next_head(self, action: int) -> tuple:
        """Where the head would end up if we took this action."""
        d = self._resolve_direction(action)
        dx, dy = d.value
        hx, hy = self.snake[0]
        return (hx + dx, hy + dy)

    def _is_collision(self, pos, ate_food=False) -> bool:
        """Check if a position is a wall or body collision."""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if pos in self.snake_set:
            # Tail will move out of the way — unless we just ate
            if not ate_food and pos == self.snake[-1]:
                return False
            return True
        return False

    # ── Food ──────────────────────────────────────────────────────────────────

    def _place_food(self) -> None:
        """Place food on a random empty cell."""
        while True:
            pos = (random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1))
            if pos not in self.snake_set:
                self.food = pos
                break

    # ── Raycasting ────────────────────────────────────────────────────────────

    def _cast_ray(self, origin, direction, body) -> tuple:
        """
        Cast a ray from origin in the given direction.
        Returns (wall_signal, body_signal) — both are 1/distance (0 if not seen).
        """
        dx, dy = direction
        step_size = (dx * dx + dy * dy) ** 0.5  # 1.0 cardinal, ~1.414 diagonal
        x, y = origin
        steps = 0
        body_signal = 0.0

        while True:
            steps += 1
            x += dx
            y += dy
            dist = steps * step_size

            if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
                return 1.0 / dist, body_signal

            if body_signal == 0.0 and (x, y) in body:
                body_signal = 1.0 / dist

    def _get_ray_dirs(self) -> list:
        """Return 8 ray directions relative to the snake's heading."""
        fx, fy = self.direction.value
        rx, ry = -fy, fx       # right
        lx, ly = fy, -fx       # left
        bx, by = -fx, -fy      # backward

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

    # ── Flood fill (BFS) ─────────────────────────────────────────────────────

    def _flood_fill(self, start, blocked, max_count, tail) -> tuple:
        """
        BFS from start, avoiding blocked cells.

        Space count caps at max_count for speed, but the search continues
        past the cap until the tail is found or the region is exhausted.

        Returns (count, found_tail).
        """
        if start in blocked:
            return 0, False

        found_tail = (start == tail)
        queue = deque([start])
        visited = {start}
        capped_count = None

        while queue:
            # Lock in the space count once we hit the cap
            if capped_count is None and len(visited) >= max_count:
                capped_count = max_count
                if found_tail:
                    return capped_count, True

            x, y = queue.popleft()

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                pos = (nx, ny)

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                # Check tail before blocked (tail may be in blocked_eat set)
                if pos == tail:
                    found_tail = True
                    if capped_count is not None:
                        return capped_count, True

                if pos in blocked or pos in visited:
                    continue

                visited.add(pos)
                queue.append(pos)

        count = capped_count if capped_count is not None else len(visited)
        return count, found_tail
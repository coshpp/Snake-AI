"""
game.py — Pygame renderer and human play loop.

Run:
  python3 game.py — default 20x20 grid
  python3 game.py <size> — size x size grid 

Keybindings:
  Q / ESC — quit
  R — reset
"""

import sys
import pygame
from snake_env import SnakeGame, Dir, CW, DEFAULT_GRID_SIZE

MAX_WINDOW = 800

C_BG = (8, 12, 16)
C_GRID = (14, 22, 30)
C_HEAD = (0, 255, 136)
C_BODY = (0, 170, 90)
C_FOOD = (255, 51, 85)
C_FOOD_GLOW = (255, 100, 120)
C_TEXT = (180, 255, 200)
C_ACCENT = (0, 200, 255)
C_DIM = (60, 100, 80)
C_PANEL_BG = (10, 16, 22)


class Renderer:
    def __init__(self, game: SnakeGame):
        pygame.init()
        self.game = game
        self._init_window()
        self.clock = pygame.time.Clock()
        self.tick = 0

        max_text_width = self.width // 5  
        approx_chars = 10  
        font_size = max(14, self.cell_size // 2)  
        self.font_sm = pygame.font.SysFont("Courier New", int(font_size))
        self.font_xs = pygame.font.SysFont("Courier New", int(font_size / 1.3))

    def _init_window(self):
        self.cell_size = max(5, min(30, MAX_WINDOW // self.game.grid_size))
        gs = self.game.grid_size
        cs = self.cell_size
        self.header_h = int(cs * 1.2)

        self.width = gs * cs
        self.height = self.header_h + gs * cs
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"SNAKE — {gs}×{gs}")

        self._glow_surfs = []
        for i in range(30):
            pulse = abs(i - 15) / 15.0
            glow_r = int(cs * 0.15 * (1 + pulse))
            surf = pygame.Surface((cs + glow_r * 2, cs + glow_r * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (*C_FOOD_GLOW, 60), surf.get_rect())
            self._glow_surfs.append((surf, glow_r))

    def resize(self):
        self._init_window()

    def draw(self, **kwargs):
        self.tick += 1
        self.screen.fill(C_BG)
        self._draw_header()
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        pygame.display.flip()
        self.clock.tick(self.game.speed)

    def draw_training_screen(self):
        self.screen.fill(C_BG)
        cx, cy = self.width // 2, self.height // 2
        text = self.font_sm.render("Training...", True, C_TEXT)
        self.screen.blit(text, (cx - text.get_width() // 2, cy - text.get_height() // 2))
        pygame.display.flip()

    def _draw_header(self):
        s = self.game
        pygame.draw.rect(self.screen, C_PANEL_BG, (0, 0, self.width, self.header_h))
        pygame.draw.line(self.screen, C_DIM, (0, self.header_h), (self.width, self.header_h), 1)

        score_surf = self.font_sm.render(f"SCORE  {s.score}", True, C_TEXT)
        record_surf = self.font_sm.render(f"RECORD  {s.record}", True, C_ACCENT)
        size_surf = self.font_sm.render(f"{s.grid_size}×{s.grid_size}", True, C_DIM)

        cy = self.header_h // 2
        pad = max(4, self.cell_size // 4)
        self.screen.blit(score_surf, (pad, cy - score_surf.get_height() // 2))
        self.screen.blit(record_surf, (self.width // 2 - record_surf.get_width() // 2,
                                       cy - record_surf.get_height() // 2))
        self.screen.blit(size_surf, (self.width - size_surf.get_width() - pad,
                                     cy - size_surf.get_height() // 2))

    def _draw_grid(self):
        gs = self.game.grid_size
        cs = self.cell_size
        for x in range(gs + 1):
            pygame.draw.line(self.screen, C_GRID, (x * cs, self.header_h), (x * cs, self.height))
        for y in range(gs + 1):
            pygame.draw.line(self.screen, C_GRID, (0, self.header_h + y * cs), (self.width, self.header_h + y * cs))

    def _draw_food(self):
        fx, fy = self.game.food
        cs = self.cell_size
        glow_surf, glow_r = self._glow_surfs[self.tick % 30]
        self.screen.blit(glow_surf, (fx * cs - glow_r, self.header_h + fy * cs - glow_r))
        pad = max(2, cs // 8)
        food_rect = pygame.Rect(fx * cs + pad, self.header_h + fy * cs + pad, cs - 2 * pad, cs - 2 * pad)
        pygame.draw.rect(self.screen, C_FOOD, food_rect, border_radius=max(2, cs // 4))

    def _draw_snake(self):
        cs = self.cell_size
        snake_list = list(self.game.snake)
        for i, (sx, sy) in enumerate(snake_list):
            r = pygame.Rect(sx * cs + 1, self.header_h + sy * cs + 1, cs - 2, cs - 2)
            if i == 0:
                pygame.draw.rect(self.screen, C_HEAD, r, border_radius=max(2, cs // 4))
                self._draw_eyes(sx, sy)
            else:
                fade = max(0.3, 1 - i / (len(snake_list) + 1))
                col = tuple(int(c * fade) for c in C_BODY)
                pygame.draw.rect(self.screen, col, r, border_radius=max(2, cs // 4))

    def _draw_eyes(self, gx, gy):
        cs = self.cell_size
        d = self.game.direction
        cx = gx * cs + cs // 2
        cy = self.header_h + gy * cs + cs // 2
        o = cs // 6
        radius = max(2, cs // 8)
        offsets = {
            Dir.RIGHT: [(o, -o), (o, o)],
            Dir.LEFT: [(-o, -o), (-o, o)],
            Dir.UP: [(-o, -o), (o, -o)],
            Dir.DOWN: [(-o, o), (o, o)],
        }
        for ox, oy in offsets[d]:
            pygame.draw.circle(self.screen, C_BG, (cx + ox, cy + oy), radius)


# ── Human play loop ───────────────────────────────────────────────────────────

def play_human(grid_size: int = DEFAULT_GRID_SIZE):
    game = SnakeGame(grid_size=grid_size, human=True)
    renderer = Renderer(game)

    dir_keys = {
        pygame.K_RIGHT: Dir.RIGHT, pygame.K_d: Dir.RIGHT,
        pygame.K_LEFT: Dir.LEFT,   pygame.K_a: Dir.LEFT,
        pygame.K_UP: Dir.UP,       pygame.K_w: Dir.UP,
        pygame.K_DOWN: Dir.DOWN,   pygame.K_s: Dir.DOWN,
    }
    opposite = {
        Dir.RIGHT: Dir.LEFT, Dir.LEFT: Dir.RIGHT,
        Dir.UP: Dir.DOWN,    Dir.DOWN: Dir.UP,
    }

    next_dir = game.direction

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    game.reset()
                    next_dir = game.direction
                if event.key in dir_keys:
                    nd = dir_keys[event.key]
                    if nd != opposite[game.direction]:
                        next_dir = nd

        cur_idx = CW.index(game.direction)
        want_idx = CW.index(next_dir)
        diff = (want_idx - cur_idx) % 4
        action = 0 if diff == 0 else (1 if diff == 1 else 2)

        _, _, done = game.step(action)
        if done:
            game.reset()
            next_dir = game.direction

        renderer.draw()


if __name__ == "__main__":
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GRID_SIZE
    play_human(grid_size=grid_size)
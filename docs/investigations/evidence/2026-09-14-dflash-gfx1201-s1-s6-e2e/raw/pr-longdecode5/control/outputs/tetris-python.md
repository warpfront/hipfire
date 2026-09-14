运行方法：安装 pygame (`pip install pygame`)，保存为 `tetris.py`，执行 `python tetris.py`。
关键设计：采用 10x20 网格，实现 7-bag 随机生成器；使用 SRS 旋转系统配合基础 Wall Kick 表；通过状态机管理游戏循环（运行/暂停/结束）；支持 Hold 槽位（每块限一次）、Next 队列（显示3个）、Ghost Piece 投影、软降/硬降、消行计分及基于消行数的等级加速。

```python
import pygame
import sys
import random
import copy

# 初始化 Pygame
pygame.init()

# 常量定义
CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
SIDE_PANEL_WIDTH = 200
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE

FPS = 60
BASE_DROP_TIME = 1000  # 毫秒
MIN_DROP_TIME = 100

# 颜色定义
COLORS = {
    'I': (0, 240, 240),
    'O': (240, 240, 0),
    'T': (160, 0, 240),
    'S': (0, 240, 0),
    'Z': (240, 0, 0),
    'J': (0, 0, 240),
    'L': (240, 160, 0),
    'BG': (20, 20, 30),
    'GRID': (50, 50, 60),
    'TEXT': (255, 255, 255),
    'GHOST': (100, 100, 100),
    'HOLD_BG': (40, 40, 50),
    'NEXT_BG': (40, 40, 50)
}

# 方块形状定义 (SRS)
SHAPES = {
    'I': [
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]
    ],
    'O': [
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]]
    ],
    'T': [
        [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    ],
    'S': [
        [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
    ],
    'Z': [
        [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ],
    'J': [
        [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 0]]
    ],
    'L': [
        [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
        [[0, 1, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    ]
}

# Wall Kick 数据 (SRS)
WALL_KICKS = {
    'JLSTZ': {
        (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)]
    },
    'I': {
        (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 1)],
        (1, 0): [(0, 0), (-1, 0), (2, 0), (-1, 1), (2, -1)],
        (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 1), (2, -1)],
        (2, 1): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 1)],
        (2, 3): [(0, 0), (1, 0), (-2, 0), (1, 1), (-2, -1)],
        (3, 2): [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 1)],
        (3, 0): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 1)],
        (0, 3): [(0, 0), (1, 0), (-2, 0), (1, 1), (-2, -1)]
    }
}

class Piece:
    def __init__(self, type, x, y, rotation=0):
        self.type = type
        self.x = x
        self.y = y
        self.rotation = rotation
        self.shape = SHAPES[type][rotation]

    def get_cells(self):
        cells = []
        for r, row in enumerate(self.shape):
            for c, val in enumerate(row):
                if val:
                    cells.append((self.x + c, self.y + r))
        return cells

    def get_ghost_y(self, board):
        ghost_y = self.y
        while True:
            if self.is_collision(board, self.x, ghost_y + 1, self.rotation):
                break
            ghost_y += 1
        return ghost_y

    def is_collision(self, board, x, y, rotation):
        shape = SHAPES[self.type][rotation]
        for r, row in enumerate(shape):
            for c, val in enumerate(row):
                if val:
                    nx, ny = x + c, y + r
                    if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                        return True
                    if ny >= 0 and board[ny][nx] != 0:
                        return True
        return False

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.reset_game()

    def reset_game(self):
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.bag = []
        self.next_queue = []
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.lines = 0
        self.level = 1
        self.drop_time = BASE_DROP_TIME
        self.last_drop_time = pygame.time.get_ticks()
        self.current_piece = self.get_next_piece()
        self.game_over = False
        self.paused = False
        self.fill_queue()

    def fill_queue(self):
        while len(self.next_queue) < 3:
            if not self.bag:
                self.bag = list('IJLOSTZ')
                random.shuffle(self.bag)
            self.next_queue.append(self.bag.pop())

    def get_next_piece(self):
        self.fill_queue()
        type = self.next_queue.pop(0)
        self.fill_queue()
        x = BOARD_WIDTH // 2 - 2
        y = 0
        return Piece(type, x, y)

    def rotate(self, direction):
        if self.current_piece.type == 'O':
            return
        old_rot = self.current_piece.rotation
        new_rot = (old_rot + direction) % 4
        if new_rot == old_rot:
            return

        kick_table = WALL_KICKS['I'] if self.current_piece.type == 'I' else WALL_KICKS['JLSTZ']
        key = (old_rot, new_rot)
        kicks = kick_table.get(key, [(0,0)])

        for dx, dy in kicks:
            new_x = self.current_piece.x + dx
            new_y = self.current_piece.y - dy
            if not self.current_piece.is_collision(self.board, new_x, new_y, new_rot):
                self.current_piece.x = new_x
                self.current_piece.y = new_y
                self.current_piece.rotation = new_rot
                break

    def move(self, dx, dy):
        new_x = self.current_piece.x + dx
        new_y = self.current_piece.y + dy
        if not self.current_piece.is_collision(self.board, new_x, new_y, self.current_piece.rotation):
            self.current_piece.x = new_x
            self.current_piece.y = new_y
            return True
        return False

    def hard_drop(self):
        while self.move(0, 1):
            self.score += 2
        self.lock_piece()

    def lock_piece(self):
        cells = self.current_piece.get_cells()
        for x, y in cells:
            if y >= 0:
                self.board[y][x] = self.current_piece.type

        self.clear_lines()
        self.current_piece = self.get_next_piece()
        self.can_hold = True

        if self.current_piece.is_collision(self.board, self.current_piece.x, self.current_piece.y, self.current_piece.rotation):
            self.game_over = True

    def clear_lines(self):
        lines_cleared = 0
        for y in range(BOARD_HEIGHT - 1, -1, -1):
            if all(cell != 0 for cell in self.board[y]):
                del self.board[y]
                self.board.insert(0, [0] * BOARD_WIDTH)
                lines_cleared += 1

        if lines_cleared > 0:
            self.lines += lines_cleared
            points = [0, 100, 300, 500, 800]
            self.score += points[lines_cleared] * self.level
            self.level = self.lines // 10 + 1
            self.drop_time = max(MIN_DROP_TIME, BASE_DROP_TIME - (self.level - 1) * 100)

    def hold(self):
        if not self.can_hold:
            return
        if self.hold_piece is None:
            self.hold_piece = self.current_piece.type
            self.current_piece = self.get_next_piece()
        else:
            temp = self.hold_piece
            self.hold_piece = self.current_piece.type
            self.current_piece = Piece(temp, BOARD_WIDTH // 2 - 2, 0)
        self.can_hold = False

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if self.game_over:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    continue

                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    continue

                if self.paused:
                    continue

                if event.key == pygame.K_LEFT:
                    self.move(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    self.move(1, 0)
                elif event.key == pygame.K_DOWN:
                    if self.move(0, 1):
                        self.score += 1
                        self.last_drop_time = pygame.time.get_ticks()
                elif event.key == pygame.K_UP:
                    self.rotate(1)
                elif event.key == pygame.K_z:
                    self.rotate(-1)
                elif event.key == pygame.K_SPACE:
                    self.hard_drop()
                elif event.key == pygame.K_c:
                    self.hold()
                elif event.key == pygame.K_r:
                    self.reset_game()

    def update(self):
        if self.game_over or self.paused:
            return

        current_time = pygame.time.get_ticks()
        if current_time - self.last_drop_time > self.drop_time:
            if not self.move(0, 1):
                self.lock_piece()
            self.last_drop_time = current_time

    def draw_cell(self, x, y, color, alpha=255):
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        if alpha < 255:
            s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.screen.blit(s, rect)
        else:
            pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw(self):
        self.screen.fill(COLORS['BG'])

        # Draw Board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] != 0:
                    self.draw_cell(x, y, COLORS[self.board[y][x]])

        # Draw Ghost
        if not self.game_over:
            ghost_y = self.current_piece.get_ghost_y(self.board)
            if ghost_y != self.current_piece.y:
                cells = []
                shape = SHAPES[self.current_piece.type][self.current_piece.rotation]
                for r, row in enumerate(shape):
                    for c, val in enumerate(row):
                        if val:
                            cells.append((self.current_piece.x + c, ghost_y + r))
                for x, y in cells:
                    if y >= 0:
                        self.draw_cell(x, y, COLORS['GHOST'], 100)

        # Draw Current Piece
        if not self.game_over:
            cells = self.current_piece.get_cells()
            for x, y in cells:
                if y >= 0:
                    self.draw_cell(x, y, COLORS[self.current_piece.type])

        # Draw Side Panel
        panel_x = BOARD_WIDTH * CELL_SIZE
        panel_y = 0

        # Hold
        pygame.draw.rect(self.screen, COLORS['HOLD_BG'], (panel_x + 10, panel_y + 10, 100, 100))
        self.screen.blit(self.font.render("HOLD", True, COLORS['TEXT']), (panel_x + 30, panel_y + 15))
        if self.hold_piece:
            shape = SHAPES[self.hold_piece][0]
            for r, row in enumerate(shape):
                for c, val in enumerate(row):
                    if val:
                        self.draw_cell(panel_x + 30 + c * 15, panel_y + 40 + r * 15, COLORS[self.hold_piece])

        # Next
        self.screen.blit(self.font.render("NEXT", True, COLORS['TEXT']), (panel_x + 30, panel_y + 120))
        for i, type in enumerate(self.next_queue):
            shape = SHAPES[type][0]
            for r, row in enumerate(shape):
                for c, val in enumerate(row):
                    if val:
                        self.draw_cell(panel_x + 30 + c * 15, panel_y + 150 + i * 40 + r * 15, COLORS[type])

        # Score
        self.screen.blit(self.font.render(f"Score: {self.score}", True, COLORS['TEXT']), (panel_x + 10, panel_y + 250))
        self.screen.blit(self.font.render(f"Lines: {self.lines}", True, COLORS['TEXT']), (panel_x + 10, panel_y + 280))
        self.screen.blit(self.font.render(f"Level: {self.level}", True, COLORS['TEXT']), (panel_x + 10, panel_y + 310))

        # Overlays
        if self.paused:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            text = self.font.render("PAUSED", True, COLORS['TEXT'])
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2))

        if self.game_over:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            text = self.font.render("GAME OVER", True, COLORS['TEXT'])
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - 20))
            text2 = self.small_font.render("Press R to Restart", True, COLORS['TEXT'])
            self.screen.blit(text2, (WINDOW_WIDTH // 2 - text2.get_width() // 2, WINDOW_HEIGHT // 2 + 20))

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == '__main__':
    game = Game()
    game.run()
```
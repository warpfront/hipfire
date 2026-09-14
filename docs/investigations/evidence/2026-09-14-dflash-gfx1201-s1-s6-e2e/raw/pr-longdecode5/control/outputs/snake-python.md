运行方法：安装 pygame (`pip install pygame`)，保存为 `snake.py`，执行 `python snake.py`。
设计：使用状态机管理游戏流程（菜单/游戏/暂停/结束），通过逻辑网格与屏幕像素分离实现窗口缩放自适应，方向队列防止反向，速度随分数动态提升。

```python
import pygame
import sys
import random

# 初始化 Pygame
pygame.init()

# 常量定义
CELL_SIZE = 20  # 逻辑单元格大小
GRID_WIDTH = 30  # 逻辑网格宽度
GRID_HEIGHT = 20  # 逻辑网格高度
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)

# 方向定义
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("贪吃蛇游戏")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)

        # 游戏状态: 'menu', 'playing', 'paused', 'game_over'
        self.state = 'menu'
        self.high_score = 0
        self.reset_game()

    def reset_game(self):
        """重置游戏状态"""
        # 蛇初始位置在中间
        self.snake = [
            (GRID_WIDTH // 2, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
            (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
        ]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.speed = 10  # 初始速度 (FPS)
        self.spawn_food()

    def spawn_food(self):
        """生成食物，确保不在蛇身上"""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def handle_events(self):
        """处理输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # 菜单状态
                if self.state == 'menu':
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        self.state = 'playing'
                        self.reset_game()

                # 游戏结束状态
                elif self.state == 'game_over':
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        self.state = 'menu'

                # 游戏中状态
                elif self.state == 'playing':
                    # 暂停/继续
                    if event.key == pygame.K_p or event.key == pygame.K_ESCAPE:
                        self.state = 'paused'
                    # 重新开始
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    # 方向控制
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        if self.direction != DOWN:
                            self.next_direction = UP
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        if self.direction != UP:
                            self.next_direction = DOWN
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        if self.direction != RIGHT:
                            self.next_direction = LEFT
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        if self.direction != LEFT:
                            self.next_direction = RIGHT

                # 暂停状态
                elif self.state == 'paused':
                    if event.key == pygame.K_p or event.key == pygame.K_ESCAPE:
                        self.state = 'playing'
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.state = 'playing'

    def update(self):
        """更新游戏逻辑"""
        if self.state != 'playing':
            return

        # 更新方向
        self.direction = self.next_direction

        # 计算新头部位置
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 碰撞检测：墙壁
        if new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT:
            self.state = 'game_over'
            if self.score > self.high_score:
                self.high_score = self.score
            return

        # 碰撞检测：自身
        if new_head in self.snake:
            self.state = 'game_over'
            if self.score > self.high_score:
                self.high_score = self.score
            return

        # 移动蛇
        self.snake.insert(0, new_head)

        # 检查是否吃到食物
        if new_head == self.food:
            self.score += 1
            # 速度随分数增加，但不超过最大值
            self.speed = min(20, 10 + self.score // 5)
            self.spawn_food()
        else:
            # 没吃到食物，移除尾部
            self.snake.pop()

    def draw_grid(self):
        """绘制背景网格"""
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                color = (240, 240, 240) if (x + y) % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_snake(self):
        """绘制蛇"""
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(self.screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # 绘制边框
            pygame.draw.rect(self.screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    def draw_food(self):
        """绘制食物"""
        x, y = self.food
        pygame.draw.rect(self.screen, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    def draw_text(self, text, font, color, x, y):
        """绘制文本辅助函数"""
        img = font.render(text, True, color)
        self.screen.blit(img, (x, y))

    def draw_ui(self):
        """绘制用户界面（分数、状态等）"""
        # 顶部信息栏
        self.draw_text(f"Score: {self.score}", self.font, BLACK, 10, 5)
        self.draw_text(f"High: {self.high_score}", self.font, BLACK, 150, 5)
        self.draw_text(f"Speed: {self.speed}", self.small_font, BLACK, 300, 5)

        if self.state == 'menu':
            self.draw_text("Press SPACE to Start", self.font, BLACK, 100, 150)
            self.draw_text("WASD/Arrows to Move", self.small_font, BLACK, 100, 200)
            self.draw_text("P/ESC to Pause", self.small_font, BLACK, 100, 230)
            self.draw_text("R to Restart", self.small_font, BLACK, 100, 260)

        elif self.state == 'paused':
            self.draw_text("PAUSED", self.font, YELLOW, 150, 150)
            self.draw_text("Press P to Resume", self.small_font, BLACK, 120, 200)

        elif self.state == 'game_over':
            self.draw_text("GAME OVER", self.font, RED, 120, 150)
            self.draw_text(f"Score: {self.score}", self.font, BLACK, 150, 200)
            self.draw_text("Press SPACE for Menu", self.small_font, BLACK, 100, 250)

    def draw(self):
        """绘制所有元素"""
        self.draw_grid()

        if self.state in ('playing', 'paused', 'game_over'):
            self.draw_snake()
            self.draw_food()
            self.draw_ui()
        else:
            self.draw_ui()

        pygame.display.flip()

    def run(self):
        """主循环"""
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.speed if self.state == 'playing' else 60)

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
```
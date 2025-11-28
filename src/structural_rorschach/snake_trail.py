"""
Snake Trail - Visualize your cave movements as a snake

Your joystick movements from the Cave game are plotted as a snake:
- UP    = snake goes up      (you went long)
- DOWN  = snake goes down    (you went short)
- LEFT  = snake goes left    (you sold)
- RIGHT = snake goes right   (you bought)

When the snake head gets close to its tail = cycle detected!
The market is returning to where it was before.

    ┌─────────────────┐
    │      LONG       │
    │        ██►      │  ← Snake head (where you are now)
    │        █        │
    │  SELL ██  BUY   │  ← Snake body (where you've been)
    │      █████      │
    │      SHORT      │
    └─────────────────┘

If head approaches tail → "I've been here before!"
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class Direction(Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)


@dataclass
class SnakeTrail:
    """
    Tracks player movements as a snake trail.

    Simple enough for an 8-year-old:
    - Snake grows as you move
    - When head gets near tail = cycle!
    """

    # The trail: list of (x, y) positions
    trail: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 0)])

    # Current position (snake head)
    x: int = 0
    y: int = 0

    # Grid bounds for display
    grid_size: int = 15

    # How close head must be to tail to trigger cycle alert
    cycle_distance: int = 2

    # Maximum trail length (snake length)
    max_length: int = 50

    def move(self, direction: Direction):
        """Add a movement to the trail."""
        dx, dy = direction.value
        self.x += dx
        self.y += dy
        self.trail.append((self.x, self.y))

        # Trim if too long
        if len(self.trail) > self.max_length:
            self.trail.pop(0)

    def move_up(self):
        self.move(Direction.UP)

    def move_down(self):
        self.move(Direction.DOWN)

    def move_left(self):
        self.move(Direction.LEFT)

    def move_right(self):
        self.move(Direction.RIGHT)

    def check_cycle(self) -> Optional[int]:
        """
        Check if head is near any part of tail.
        Returns distance to nearest tail segment, or None if no cycle.
        """
        if len(self.trail) < 5:
            return None

        head = self.trail[-1]

        # Check against older parts of trail (not recent moves)
        for i, pos in enumerate(self.trail[:-5]):
            dist = abs(head[0] - pos[0]) + abs(head[1] - pos[1])
            if dist <= self.cycle_distance:
                return dist

        return None

    def render(self, width: int = 17, height: int = 11) -> str:
        """
        Render snake trail as ASCII.

        ┌───────────────┐
        │     LONG      │
        │       █►      │
        │       █       │
        │ SELL ██  BUY  │
        │     SHORT     │
        └───────────────┘
        """
        lines = []
        inner_w = width - 2
        inner_h = height - 2

        # Create grid
        grid = [[' ' for _ in range(inner_w)] for _ in range(inner_h)]

        # Center of grid
        cx, cy = inner_w // 2, inner_h // 2

        # Plot trail
        for i, (tx, ty) in enumerate(self.trail):
            # Convert to grid coordinates (centered)
            gx = cx + tx
            gy = cy - ty  # Flip y so UP is up on screen

            if 0 <= gx < inner_w and 0 <= gy < inner_h:
                if i == len(self.trail) - 1:
                    # Head
                    grid[gy][gx] = '►'
                elif i == 0:
                    # Tail
                    grid[gy][gx] = '○'
                else:
                    # Body
                    grid[gy][gx] = '█'

        # Build output
        lines.append("┌" + "─" * inner_w + "┐")

        # Add LONG label at top
        long_label = "LONG"
        long_pad = (inner_w - len(long_label)) // 2
        lines.append("│" + " " * long_pad + long_label + " " * (inner_w - long_pad - len(long_label)) + "│")

        # Grid rows (skip first and last for labels)
        for row in range(1, inner_h - 1):
            # Add SELL/BUY labels in middle row
            if row == inner_h // 2:
                row_str = "".join(grid[row])
                # Try to fit SELL on left, BUY on right
                left = "SELL "
                right = " BUY"
                mid_content = row_str[len(left):inner_w - len(right)]
                lines.append("│" + left + mid_content + right + "│")
            else:
                lines.append("│" + "".join(grid[row]) + "│")

        # Add SHORT label at bottom
        short_label = "SHORT"
        short_pad = (inner_w - len(short_label)) // 2
        lines.append("│" + " " * short_pad + short_label + " " * (inner_w - short_pad - len(short_label)) + "│")

        lines.append("└" + "─" * inner_w + "┘")

        # Cycle detection
        cycle_dist = self.check_cycle()
        if cycle_dist is not None:
            lines.append(f"  !! CYCLE DETECTED !!")

        return "\n".join(lines)


class CaveWithSnake:
    """
    Combines Cave game with Snake trail visualization.

    Two views side by side:
    - Left: The cave you're flying through
    - Right: Your movement history as a snake
    """

    def __init__(self):
        # Import here to avoid circular dependency
        from .simple_cave import SimpleCaveGame

        self.cave = SimpleCaveGame(cave_width=30)
        self.snake = SnakeTrail()

        # Track last input for snake
        self.last_direction = Direction.NONE

    def input_up(self):
        self.cave.input_up()
        self.snake.move_up()
        self.last_direction = Direction.UP

    def input_down(self):
        self.cave.input_down()
        self.snake.move_down()
        self.last_direction = Direction.DOWN

    def input_left(self):
        self.cave.input_left()
        self.snake.move_left()
        self.last_direction = Direction.LEFT

    def input_right(self):
        self.cave.input_right()
        self.snake.move_right()
        self.last_direction = Direction.RIGHT

    def tick(self):
        self.cave.generate_random_cave()
        return self.cave.tick()

    def render(self) -> str:
        """Render both views side by side."""
        cave_lines = self.cave.render(height=11, width=30).split("\n")
        snake_lines = self.snake.render(width=17, height=11).split("\n")

        # Combine side by side
        combined = []
        max_lines = max(len(cave_lines), len(snake_lines))

        for i in range(max_lines):
            cave_line = cave_lines[i] if i < len(cave_lines) else " " * 30
            snake_line = snake_lines[i] if i < len(snake_lines) else ""
            combined.append(f"{cave_line}  {snake_line}")

        return "\n".join(combined)

    def reset(self):
        self.cave.reset()
        self.snake = SnakeTrail()


def demo():
    """Demo the combined cave + snake view."""
    print("Cave + Snake Trail Demo")
    print("=" * 50)
    print()
    print("Left: Fly through the cave")
    print("Right: Your movement history as a snake")
    print()

    # Demo just the snake trail first
    snake = SnakeTrail()
    print("Snake trail only:")
    print(snake.render())
    print()

    # Make some moves
    moves = ["UP", "RIGHT", "RIGHT", "UP", "UP", "LEFT", "DOWN", "DOWN", "DOWN", "LEFT"]
    for move in moves:
        if move == "UP":
            snake.move_up()
        elif move == "DOWN":
            snake.move_down()
        elif move == "LEFT":
            snake.move_left()
        elif move == "RIGHT":
            snake.move_right()

    print(f"After moves: {', '.join(moves)}")
    print(snake.render())
    print()

    # Try to create a cycle
    print("Continuing to create a cycle...")
    for move in ["LEFT", "UP", "UP", "RIGHT", "RIGHT"]:
        if move == "UP":
            snake.move_up()
        elif move == "DOWN":
            snake.move_down()
        elif move == "LEFT":
            snake.move_left()
        elif move == "RIGHT":
            snake.move_right()

    print(snake.render())
    return

    # Full game demo (only works when imported as module)
    game = CaveWithSnake()

    print("Initial:")
    print(game.render())
    print()

    # Make some moves
    moves = [
        ("UP", game.input_up),
        ("RIGHT", game.input_right),
        ("RIGHT", game.input_right),
        ("UP", game.input_up),
        ("UP", game.input_up),
        ("LEFT", game.input_left),
        ("DOWN", game.input_down),
        ("DOWN", game.input_down),
        ("DOWN", game.input_down),
        ("LEFT", game.input_left),
    ]

    for name, move in moves:
        move()
        game.tick()

    print(f"After moves (UP, RIGHT, RIGHT, UP, UP, LEFT, DOWN, DOWN, DOWN, LEFT):")
    print(game.render())
    print()

    # Continue to create a cycle
    print("Creating a cycle...")
    for name, move in [("LEFT", game.input_left), ("UP", game.input_up), ("UP", game.input_up)]:
        move()
        game.tick()

    print(game.render())


if __name__ == "__main__":
    demo()

"""
Simple Cave - Nokia-era simplicity with 2D controls

A rocket flies through a cave. Don't hit the walls.
So simple an 8-year-old can play it.

Controls:
  UP    = go up      (secretly: go long)
  DOWN  = go down    (secretly: go short)
  LEFT  = go left    (secretly: sell)
  RIGHT = go right   (secretly: buy)

The 4 cave walls come from real market data:
  - Floor   = Price + Support
  - Ceiling = Volume + Resistance
  - Left    = Bid depth
  - Right   = Ask depth

But the player just sees a cave and tries to survive.

    ═══════════════════════════════════
    ▌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▐   4 WALLS
    ▌                                ▐
    ▌            ◆                   ▐   ROCKET
    ▌                                ▐
    ▌▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▐
    ═══════════════════════════════════
    SCORE: 42

What the player sees:     What's really happening:
        UP                    LONG ▲
         │                       │
  LEFT ──┼── RIGHT        SELL ──┼── BUY
         │                       │
       DOWN                  SHORT ▼
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import random


class GameState(Enum):
    PLAYING = "playing"
    CRASHED = "crashed"
    PAUSED = "paused"


@dataclass
class CaveSlice:
    """One vertical slice of the cave with 4 walls."""
    floor: float     # 0.0 to 1.0 (price + support)
    ceiling: float   # 0.0 to 1.0 (volume + resistance)
    left: float      # 0.0 to 1.0 (bid wall position)
    right: float     # 0.0 to 1.0 (ask wall position)


@dataclass
class Rocket:
    """The player's rocket in 2D space."""
    x: float = 0.5   # Horizontal position (0=left wall, 1=right wall)
    y: float = 0.5   # Vertical position (0=floor, 1=ceiling)
    vx: float = 0    # Horizontal velocity
    vy: float = 0    # Vertical velocity


@dataclass
class Game:
    """Complete game state."""
    rocket: Rocket
    cave: List[CaveSlice]
    score: int = 0
    state: GameState = GameState.PLAYING
    high_score: int = 0


class SimpleCaveGame:
    """
    Simple cave game with full 2D controls.

    Player navigates with joystick in 4 directions.
    Cave walls come from market data (invisibly).
    """

    def __init__(
        self,
        cave_width: int = 40,       # How many slices visible
        gravity: float = 0.015,     # Downward pull
        drift: float = 0.01,        # Leftward pull (time pressure)
        thrust: float = 0.04,       # Movement per input
        min_gap_v: float = 0.3,     # Minimum vertical gap
        min_gap_h: float = 0.3,     # Minimum horizontal gap
    ):
        self.cave_width = cave_width
        self.gravity = gravity
        self.drift = drift
        self.thrust = thrust
        self.min_gap_v = min_gap_v
        self.min_gap_h = min_gap_h

        self.game = Game(
            rocket=Rocket(),
            cave=[],
        )

        # Initialize with open cave
        for _ in range(cave_width):
            self.game.cave.append(CaveSlice(
                floor=0.15,
                ceiling=0.85,
                left=0.1,
                right=0.9
            ))

    def feed_market_bar(
        self,
        price: float,
        volume: float,
        bid_depth: float,
        ask_depth: float,
        price_min: float,
        price_max: float,
        volume_max: float,
        depth_max: float = 1.0
    ):
        """
        Feed one bar of market data to generate cave slice.

        Args:
            price: Current price → floor height
            volume: Current volume → ceiling pressure
            bid_depth: Order book bid depth → left wall
            ask_depth: Order book ask depth → right wall
            price_min/max: For normalization
            volume_max: For normalization
            depth_max: For normalization
        """
        # Price → floor (higher price = higher floor)
        price_range = price_max - price_min if price_max > price_min else 1
        floor = 0.1 + 0.35 * ((price - price_min) / price_range)

        # Volume → ceiling (higher volume = lower ceiling)
        vol_norm = min(1.0, volume / volume_max) if volume_max > 0 else 0.5
        ceiling = 0.9 - 0.35 * vol_norm

        # Bid depth → left wall (thicker bids = wall pushes right)
        bid_norm = min(1.0, bid_depth / depth_max) if depth_max > 0 else 0.3
        left = 0.05 + 0.25 * bid_norm

        # Ask depth → right wall (thicker asks = wall pushes left)
        ask_norm = min(1.0, ask_depth / depth_max) if depth_max > 0 else 0.3
        right = 0.95 - 0.25 * ask_norm

        # Ensure minimum gaps
        if ceiling - floor < self.min_gap_v:
            mid = (ceiling + floor) / 2
            floor = mid - self.min_gap_v / 2
            ceiling = mid + self.min_gap_v / 2

        if right - left < self.min_gap_h:
            mid = (right + left) / 2
            left = mid - self.min_gap_h / 2
            right = mid + self.min_gap_h / 2

        # Scroll cave
        self.game.cave.pop(0)
        self.game.cave.append(CaveSlice(
            floor=floor,
            ceiling=ceiling,
            left=left,
            right=right
        ))

    def generate_random_cave(self):
        """Generate cave without market data."""
        if self.game.cave:
            last = self.game.cave[-1]
            # Gentle random walk for all 4 walls
            new_floor = last.floor + random.uniform(-0.03, 0.03)
            new_ceiling = last.ceiling + random.uniform(-0.03, 0.03)
            new_left = last.left + random.uniform(-0.02, 0.02)
            new_right = last.right + random.uniform(-0.02, 0.02)
        else:
            new_floor, new_ceiling = 0.15, 0.85
            new_left, new_right = 0.1, 0.9

        # Clamp
        new_floor = max(0.05, min(0.4, new_floor))
        new_ceiling = max(0.6, min(0.95, new_ceiling))
        new_left = max(0.05, min(0.35, new_left))
        new_right = max(0.65, min(0.95, new_right))

        # Ensure gaps
        if new_ceiling - new_floor < self.min_gap_v:
            mid = (new_ceiling + new_floor) / 2
            new_floor = mid - self.min_gap_v / 2
            new_ceiling = mid + self.min_gap_v / 2

        if new_right - new_left < self.min_gap_h:
            mid = (new_right + new_left) / 2
            new_left = mid - self.min_gap_h / 2
            new_right = mid + self.min_gap_h / 2

        self.game.cave.pop(0)
        self.game.cave.append(CaveSlice(
            floor=new_floor,
            ceiling=new_ceiling,
            left=new_left,
            right=new_right
        ))

    def input_up(self):
        """Player pressed UP (secretly: go long)."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.vy += self.thrust

    def input_down(self):
        """Player pressed DOWN (secretly: go short)."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.vy -= self.thrust

    def input_left(self):
        """Player pressed LEFT (secretly: sell)."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.vx -= self.thrust

    def input_right(self):
        """Player pressed RIGHT (secretly: buy)."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.vx += self.thrust

    def tick(self) -> bool:
        """
        Advance game by one frame.
        Returns True if still playing, False if crashed.
        """
        if self.game.state != GameState.PLAYING:
            return False

        # Apply gravity (pulls down) and drift (pulls left)
        self.game.rocket.vy -= self.gravity
        self.game.rocket.vx -= self.drift

        # Dampen velocity
        self.game.rocket.vx *= 0.92
        self.game.rocket.vy *= 0.92

        # Move rocket
        self.game.rocket.x += self.game.rocket.vx
        self.game.rocket.y += self.game.rocket.vy

        # Check collision
        rocket_slice_idx = 5  # Rocket position in cave view
        if rocket_slice_idx < len(self.game.cave):
            s = self.game.cave[rocket_slice_idx]

            # Check all 4 walls
            crashed = (
                self.game.rocket.y <= s.floor or      # Hit floor
                self.game.rocket.y >= s.ceiling or    # Hit ceiling
                self.game.rocket.x <= s.left or       # Hit left wall
                self.game.rocket.x >= s.right         # Hit right wall
            )

            if crashed:
                self.game.state = GameState.CRASHED
                if self.game.score > self.game.high_score:
                    self.game.high_score = self.game.score
                return False

        # Still alive
        self.game.score += 1
        return True

    def reset(self):
        """Start new game."""
        self.game.rocket = Rocket()
        self.game.score = 0
        self.game.state = GameState.PLAYING

    def render(self, height: int = 12, width: int = 40) -> str:
        """
        Render game as ASCII art with 4 walls visible.

        ═══════════════════════════════════════
        ▌▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▐
        ▌                                    ▐
        ▌              ◆                     ▐
        ▌                                    ▐
        ▌▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▐
        ═══════════════════════════════════════
        SCORE: 42                    HIGH: 891
        """
        lines = []

        # Top border
        lines.append("═" * width)

        rocket_col = 5  # Where rocket appears horizontally in view
        rocket_slice = self.game.cave[rocket_col] if rocket_col < len(self.game.cave) else None

        # Calculate rocket screen position
        rocket_row = int((1 - self.game.rocket.y) * (height - 1))
        rocket_row = max(0, min(height - 1, rocket_row))

        # Render each row
        for row in range(height):
            row_normalized = 1 - (row / (height - 1))  # 1.0 at top, 0.0 at bottom

            line_chars = []
            for col in range(width):
                slice_idx = col
                if slice_idx < len(self.game.cave):
                    s = self.game.cave[slice_idx]

                    # Determine what character to draw
                    # First check if we're in the left/right wall zones
                    col_normalized = col / (width - 1)  # 0 to 1

                    # Map slice walls to screen columns
                    left_wall_col = int(s.left * width)
                    right_wall_col = int(s.right * width)

                    in_left_wall = col <= left_wall_col
                    in_right_wall = col >= right_wall_col
                    in_ceiling = row_normalized >= s.ceiling
                    in_floor = row_normalized <= s.floor

                    # Is this the rocket position?
                    is_rocket = (col == rocket_col and row == rocket_row)

                    if is_rocket:
                        if self.game.state == GameState.CRASHED:
                            char = "✖"
                        else:
                            char = "◆"
                    elif in_left_wall:
                        if in_ceiling:
                            char = "▛"
                        elif in_floor:
                            char = "▙"
                        else:
                            char = "▌"
                    elif in_right_wall:
                        if in_ceiling:
                            char = "▜"
                        elif in_floor:
                            char = "▟"
                        else:
                            char = "▐"
                    elif in_ceiling:
                        char = "▄"
                    elif in_floor:
                        char = "▀"
                    else:
                        char = " "
                else:
                    char = " "

                line_chars.append(char)

            lines.append("".join(line_chars))

        # Bottom border
        lines.append("═" * width)

        # Score line
        score_str = f"SCORE: {self.game.score}"
        high_str = f"HIGH: {self.game.high_score}"
        padding = width - len(score_str) - len(high_str)
        lines.append(f"{score_str}{' ' * max(1, padding)}{high_str}")

        if self.game.state == GameState.CRASHED:
            crash_msg = "CRASHED! Press R to restart"
            pad = (width - len(crash_msg)) // 2
            lines.append(" " * pad + crash_msg)

        return "\n".join(lines)


def demo():
    """Demo the 2D cave game."""
    print("Simple Cave Game - 2D Controls Demo")
    print("=" * 40)
    print()
    print("Controls:")
    print("  UP/DOWN   = Vertical movement")
    print("  LEFT/RIGHT = Horizontal movement")
    print()
    print("(Secretly: UP=Long, DOWN=Short, LEFT=Sell, RIGHT=Buy)")
    print()

    game = SimpleCaveGame(cave_width=40)

    print("Initial state:")
    print(game.render())
    print()

    # Move in all 4 directions
    for _ in range(3):
        game.input_up()
        game.input_right()
        game.generate_random_cave()
        game.tick()

    print("After UP+RIGHT (secretly: going long + buying):")
    print(game.render())
    print()

    # Now go down and left
    for _ in range(5):
        game.input_down()
        game.input_left()
        game.generate_random_cave()
        game.tick()

    print("After DOWN+LEFT (secretly: going short + selling):")
    print(game.render())


if __name__ == "__main__":
    demo()

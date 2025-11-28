"""
Simple Cave - Nokia-era simplicity

A rocket flies through a cave. Don't hit the walls.
So simple an 8-year-old can play it.

Controls:
  UP    = go up
  DOWN  = go down
  That's it.

The cave walls come from real market data (price = floor, volume = ceiling),
but the player doesn't need to know that. They just see a cave.

Inspired by:
- Snake (Nokia)
- Helicopter Game
- Flappy Bird

    ════════════════════════════════
         ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄    CEILING

                 ◆────>     ROCKET

    ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀     FLOOR
    ════════════════════════════════

    SCORE: 42
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import random


class GameState(Enum):
    PLAYING = "playing"
    CRASHED = "crashed"
    PAUSED = "paused"


@dataclass
class CaveSlice:
    """One vertical slice of the cave."""
    floor: float    # 0.0 to 1.0 (bottom of screen)
    ceiling: float  # 0.0 to 1.0 (top of screen)

    @property
    def gap(self) -> float:
        """How much space between floor and ceiling."""
        return self.ceiling - self.floor


@dataclass
class Rocket:
    """The player's rocket."""
    y: float = 0.5       # Vertical position (0=bottom, 1=top)
    velocity: float = 0  # Current vertical speed


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
    The simplest possible cave game.

    Market data is converted to cave geometry invisibly.
    Player just flies and avoids walls.
    """

    def __init__(
        self,
        cave_width: int = 40,      # How many slices visible
        gravity: float = 0.02,      # How fast rocket falls
        thrust: float = 0.05,       # How much UP adds
        min_gap: float = 0.25,      # Minimum cave opening
    ):
        self.cave_width = cave_width
        self.gravity = gravity
        self.thrust = thrust
        self.min_gap = min_gap

        self.game = Game(
            rocket=Rocket(),
            cave=[],
        )

        # Initialize with flat cave
        for _ in range(cave_width):
            self.game.cave.append(CaveSlice(floor=0.2, ceiling=0.8))

    def feed_market_bar(self, price: float, volume: float,
                        price_min: float, price_max: float,
                        volume_max: float):
        """
        Feed one bar of market data to generate cave.

        The player never sees these numbers - just the resulting cave shape.
        """
        # Price → floor height (higher price = higher floor)
        price_range = price_max - price_min if price_max > price_min else 1
        floor = 0.1 + 0.4 * ((price - price_min) / price_range)

        # Volume → ceiling pressure (higher volume = lower ceiling)
        vol_normalized = min(1.0, volume / volume_max) if volume_max > 0 else 0.5
        ceiling_drop = 0.3 * vol_normalized
        ceiling = 0.9 - ceiling_drop

        # Ensure minimum gap
        if ceiling - floor < self.min_gap:
            mid = (ceiling + floor) / 2
            floor = mid - self.min_gap / 2
            ceiling = mid + self.min_gap / 2

        # Scroll cave left, add new slice on right
        self.game.cave.pop(0)
        self.game.cave.append(CaveSlice(floor=floor, ceiling=ceiling))

    def generate_random_cave(self):
        """Generate cave without market data (for pure game mode)."""
        if self.game.cave:
            last = self.game.cave[-1]
            # Gentle random walk
            new_floor = last.floor + random.uniform(-0.05, 0.05)
            new_ceiling = last.ceiling + random.uniform(-0.05, 0.05)
        else:
            new_floor = 0.2
            new_ceiling = 0.8

        # Clamp to valid range
        new_floor = max(0.05, min(0.45, new_floor))
        new_ceiling = max(0.55, min(0.95, new_ceiling))

        # Ensure gap
        if new_ceiling - new_floor < self.min_gap:
            mid = (new_ceiling + new_floor) / 2
            new_floor = mid - self.min_gap / 2
            new_ceiling = mid + self.min_gap / 2

        self.game.cave.pop(0)
        self.game.cave.append(CaveSlice(floor=new_floor, ceiling=new_ceiling))

    def input_up(self):
        """Player pressed UP."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.velocity += self.thrust

    def input_down(self):
        """Player pressed DOWN."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.velocity -= self.thrust

    def tick(self) -> bool:
        """
        Advance game by one frame.
        Returns True if still playing, False if crashed.
        """
        if self.game.state != GameState.PLAYING:
            return False

        # Apply gravity
        self.game.rocket.velocity -= self.gravity

        # Dampen velocity
        self.game.rocket.velocity *= 0.95

        # Move rocket
        self.game.rocket.y += self.game.rocket.velocity

        # Check collision with current slice (rocket is at left side of screen)
        rocket_slice_idx = 5  # Rocket is at position 5 in the cave
        if rocket_slice_idx < len(self.game.cave):
            slice = self.game.cave[rocket_slice_idx]

            if self.game.rocket.y <= slice.floor or self.game.rocket.y >= slice.ceiling:
                self.game.state = GameState.CRASHED
                if self.game.score > self.game.high_score:
                    self.game.high_score = self.game.score
                return False

        # Still alive - increment score
        self.game.score += 1
        return True

    def reset(self):
        """Start a new game."""
        self.game.rocket = Rocket()
        self.game.score = 0
        self.game.state = GameState.PLAYING
        # Keep cave as is - continuous experience

    def render(self, height: int = 12) -> str:
        """
        Render game as ASCII art.

        Example:
        ════════════════════════════════════════
        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
                           ▄▄▄▄▄
                   ▄▄▄▄▄▄▄▄     ▄▄▄▄
             ◆──>
         ▀▀▀▀▀▀▀▀▀▀▀▀     ▀▀▀▀▀
        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
        ════════════════════════════════════════
        SCORE: 142                    HIGH: 891
        """
        lines = []
        width = self.cave_width

        # Top border
        lines.append("═" * width)

        # Build cave view row by row
        rocket_x = 5  # Rocket horizontal position
        rocket_y_row = int((1 - self.game.rocket.y) * (height - 1))
        rocket_y_row = max(0, min(height - 1, rocket_y_row))

        for row in range(height):
            row_normalized = 1 - (row / (height - 1))  # 1.0 at top, 0.0 at bottom

            line_chars = []
            for col in range(width):
                if col < len(self.game.cave):
                    slice = self.game.cave[col]

                    # Is this position inside the cave or wall?
                    if row_normalized >= slice.ceiling:
                        # Ceiling (wall)
                        char = "▄"
                    elif row_normalized <= slice.floor:
                        # Floor (wall)
                        char = "▀"
                    elif col == rocket_x and row == rocket_y_row:
                        # Rocket!
                        if self.game.state == GameState.CRASHED:
                            char = "✖"
                        else:
                            char = "◆"
                    else:
                        # Empty space
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
        lines.append(f"{score_str}{' ' * padding}{high_str}")

        if self.game.state == GameState.CRASHED:
            crash_msg = "CRASHED! Press R to restart"
            pad = (width - len(crash_msg)) // 2
            lines.append(" " * pad + crash_msg)

        return "\n".join(lines)


def demo():
    """Quick demo of the simple cave game."""
    print("Simple Cave Game Demo")
    print("=" * 40)
    print()

    game = SimpleCaveGame(cave_width=40)

    # Simulate a few frames with random cave
    print("Initial state:")
    print(game.render())
    print()

    # Player goes up a few times
    for _ in range(5):
        game.input_up()
        game.generate_random_cave()
        game.tick()

    print("After going UP:")
    print(game.render())
    print()

    # Let gravity take over
    for _ in range(20):
        game.generate_random_cave()
        game.tick()

    print("After drifting:")
    print(game.render())

    print()
    print("Controls: UP to thrust up, DOWN to dive")
    print("That's it. Just don't hit the walls.")


if __name__ == "__main__":
    demo()

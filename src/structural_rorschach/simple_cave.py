"""
Simple Cave - Nokia-era simplicity with 2D controls

A rocket flies through a cave. Don't hit the walls.
So simple an 8-year-old can play it.

CONTROLS (what the kid sees → what it means for trading):
  ↑ UP    = go up      → LONG  (bet price goes higher)
  ↓ DOWN  = go down    → SHORT (bet price goes lower)
  ← LEFT  = go left    → SELL  (exit position, consume bids)
  → RIGHT = go right   → BUY   (enter position, consume asks)

THE 4 WALLS (from real market data):

    ════════════════════════════════════════════
    ▌▄▄▄▄▄▄▄▄ CEILING (Volume) ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▐
    ▌   High volume = ceiling drops              ▐
    ▌   (resistance, hard to push through)       ▐
    ▌                                            ▐
    L                      ◆                     R
    E   ←── Bids support         Asks resist ──→ I
    F       you if you fall      if you push     G
    T                                            H
    ▌                                            T
    ▌▀▀▀▀▀▀▀▀ FLOOR (Price) ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▐
    ▌   Higher price = floor rises               ▐
    ════════════════════════════════════════════

WALL MEANINGS:
  FLOOR (bottom)  = Current Price - Higher price raises the floor
  CEILING (top)   = Volume        - High volume compresses space (resistance)
  LEFT WALL       = Bid Depth     - More buyers = wall pushes right (support)
  RIGHT WALL      = Ask Depth     - More sellers = wall pushes left (resistance)

QUADRANTS (trading meaning):
  TOP-RIGHT    = LONG + BUY   = Opening a long position ✓
  BOTTOM-LEFT  = SHORT + SELL = Closing/Opening short  ✓
  TOP-LEFT     = LONG + SELL  = Closing a long position
  BOTTOM-RIGHT = SHORT + BUY  = Closing a short position

SAFE ZONE: Stay in the open space!
  - Above floor = not underwater on your position
  - Below ceiling = not buying into a volume spike
  - Away from walls = liquidity to enter/exit
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
        gravity: float = 0.0,       # No gravity (trading is neutral)
        drift: float = 0.0,         # No drift (simpler for kids)
        step_size: float = 0.05,    # Movement per input (direct, no momentum)
        min_gap_v: float = 0.4,     # Minimum vertical gap (more room)
        min_gap_h: float = 0.4,     # Minimum horizontal gap (more room)
    ):
        self.cave_width = cave_width
        self.gravity = gravity
        self.drift = drift
        self.step_size = step_size
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
        """Player pressed UP (secretly: go long). Direct position control."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.y = min(1.0, self.game.rocket.y + self.step_size)

    def input_down(self):
        """Player pressed DOWN (secretly: go short). Direct position control."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.y = max(0.0, self.game.rocket.y - self.step_size)

    def input_left(self):
        """Player pressed LEFT (secretly: sell). Direct position control."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.x = max(0.0, self.game.rocket.x - self.step_size)

    def input_right(self):
        """Player pressed RIGHT (secretly: buy). Direct position control."""
        if self.game.state == GameState.PLAYING:
            self.game.rocket.x = min(1.0, self.game.rocket.x + self.step_size)

    def tick(self) -> bool:
        """
        Advance game by one frame.
        Returns True if still playing, False if crashed.
        """
        if self.game.state != GameState.PLAYING:
            return False

        # No momentum - rocket stays where you put it (Nokia-style direct control)

        # Check collision - use center slice since walls vary across screen
        rocket_slice_idx = self.cave_width // 2
        if rocket_slice_idx < len(self.game.cave):
            s = self.game.cave[rocket_slice_idx]

            # Check all 4 walls (with small margin for fairness)
            margin = 0.02
            crashed = (
                self.game.rocket.y <= s.floor + margin or      # Hit floor
                self.game.rocket.y >= s.ceiling - margin or    # Hit ceiling
                self.game.rocket.x <= s.left + margin or       # Hit left wall
                self.game.rocket.x >= s.right - margin         # Hit right wall
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

    def render(self, height: int = 12, width: int = 40, show_quadrants: bool = True) -> str:
        """
        Render game as ASCII art with 4 walls visible.

        With quadrants showing trading meaning:

        ═══════════════════════════════════════
        ▌▄▄▄▄▄▄▄(VOLUME/RESISTANCE)▄▄▄▄▄▄▄▄▄▄▐
        ▌  LONG+SELL  │  LONG+BUY           ▐
        ▌  (close     │  (open              ▐
        ▌   long)     │   long)             ▐
        B─────────────┼───────────────────────A
        I  SHORT+SELL │  SHORT+BUY          S
        D  (open      │  (close             K
        S   short)    │   short)            S
        ▌▀▀▀▀▀▀▀▀(PRICE/SUPPORT)▀▀▀▀▀▀▀▀▀▀▀▀▐
        ═══════════════════════════════════════
        SCORE: 42                    HIGH: 891
        """
        lines = []

        # Top border
        lines.append("═" * width)

        # Rocket position on screen - map rocket.x (0-1) to screen column
        rocket_col = int(self.game.rocket.x * (width - 1))
        rocket_col = max(0, min(width - 1, rocket_col))

        # Calculate rocket screen row - map rocket.y (0-1) to screen row
        rocket_row = int((1 - self.game.rocket.y) * (height - 1))
        rocket_row = max(0, min(height - 1, rocket_row))

        # Use center slice for wall rendering (the cave scrolls past)
        center_slice_idx = width // 2

        # Render each row
        for row in range(height):
            row_normalized = 1 - (row / (height - 1))  # 1.0 at top, 0.0 at bottom

            line_chars = []
            for col in range(width):
                slice_idx = col
                if slice_idx < len(self.game.cave):
                    s = self.game.cave[slice_idx]

                    # Map slice walls to screen columns
                    left_wall_col = int(s.left * (width - 1))
                    right_wall_col = int(s.right * (width - 1))

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

        # Add quadrant legend if enabled
        if show_quadrants:
            lines.append("")
            lines.append("─" * width)
            legend = "↑LONG  ↓SHORT  ←SELL  →BUY"
            pad = (width - len(legend)) // 2
            lines.append(" " * pad + legend)
            walls = "CEIL=Volume  FLOOR=Price  L=Bids  R=Asks"
            pad2 = (width - len(walls)) // 2
            lines.append(" " * pad2 + walls)

        return "\n".join(lines)


def render_vertical_history(
    cave_history: List[CaveSlice],
    rocket_y_history: List[float],
    height: int = 12,
    width: int = 20
) -> str:
    """
    Render vertical (up/down) historical view.

    Shows the last N values of floor and ceiling, with rocket Y positions.

    Args:
        cave_history: List of CaveSlice (most recent last)
        rocket_y_history: List of rocket Y positions (most recent last)
        height: Render height
        width: Number of historical points to show

    Returns:
        ASCII art string showing floor, ceiling, and rocket path
    """
    lines = []
    lines.append("UP/DOWN History (last {} bars)".format(width))
    lines.append("─" * width)

    # Use last 'width' entries
    caves = cave_history[-width:] if len(cave_history) >= width else cave_history
    rockets = rocket_y_history[-width:] if len(rocket_y_history) >= width else rocket_y_history

    # Pad if needed
    while len(caves) < width:
        caves.insert(0, CaveSlice(floor=0.15, ceiling=0.85, left=0.1, right=0.9))
    while len(rockets) < width:
        rockets.insert(0, 0.5)

    # Render each row (top to bottom = y=1 to y=0)
    for row in range(height):
        y_val = 1.0 - (row / (height - 1))  # 1.0 at top, 0.0 at bottom

        row_chars = []
        for col in range(width):
            cave = caves[col]
            rocket_y = rockets[col]

            # Determine character
            rocket_row = int((1 - rocket_y) * (height - 1))
            is_rocket = (row == rocket_row)

            in_ceiling = y_val >= cave.ceiling
            in_floor = y_val <= cave.floor

            if is_rocket:
                char = "●"  # Rocket position
            elif in_ceiling:
                char = "▄"  # Ceiling
            elif in_floor:
                char = "▀"  # Floor
            else:
                char = " "  # Open space

            row_chars.append(char)

        lines.append("".join(row_chars))

    lines.append("─" * width)
    lines.append("OLD" + " " * (width - 6) + "NEW")

    return "\n".join(lines)


def render_horizontal_history(
    cave_history: List[CaveSlice],
    rocket_x_history: List[float],
    height: int = 8,
    width: int = 20
) -> str:
    """
    Render horizontal (left/right) historical view.

    Shows the last N values of left and right walls, with rocket X positions.

    Args:
        cave_history: List of CaveSlice (most recent last)
        rocket_x_history: List of rocket X positions (most recent last)
        height: Render height (represents left-right space)
        width: Number of historical points to show

    Returns:
        ASCII art string showing left wall, right wall, and rocket path
    """
    lines = []
    lines.append("LEFT/RIGHT History (last {} bars)".format(width))
    lines.append("─" * width)

    # Use last 'width' entries
    caves = cave_history[-width:] if len(cave_history) >= width else cave_history
    rockets = rocket_x_history[-width:] if len(rocket_x_history) >= width else rocket_x_history

    # Pad if needed
    while len(caves) < width:
        caves.insert(0, CaveSlice(floor=0.15, ceiling=0.85, left=0.1, right=0.9))
    while len(rockets) < width:
        rockets.insert(0, 0.5)

    # Render each row (top to bottom = x=1 (right) to x=0 (left))
    for row in range(height):
        x_val = 1.0 - (row / (height - 1))  # 1.0 at top (right side), 0.0 at bottom (left side)

        row_chars = []
        for col in range(width):
            cave = caves[col]
            rocket_x = rockets[col]

            # Determine character
            rocket_row = int((1 - rocket_x) * (height - 1))
            is_rocket = (row == rocket_row)

            in_right = x_val >= cave.right
            in_left = x_val <= cave.left

            if is_rocket:
                char = "●"  # Rocket position
            elif in_right:
                char = "▐"  # Right wall
            elif in_left:
                char = "▌"  # Left wall
            else:
                char = " "  # Open space

            row_chars.append(char)

        lines.append("".join(row_chars))

    lines.append("─" * width)
    # Labels: RIGHT at top, LEFT at bottom
    lines.insert(2, "R")  # Right label at top
    lines.append("L" + " " * (width - 1))  # Left label at bottom
    lines.append("OLD" + " " * (width - 6) + "NEW")

    return "\n".join(lines)


class HistoryTracker:
    """
    Tracks cave and rocket position history for rendering plots.

    Keeps a rolling window of the last N states.
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.cave_history: List[CaveSlice] = []
        self.rocket_y_history: List[float] = []
        self.rocket_x_history: List[float] = []

    def record(self, cave_slice: CaveSlice, rocket_x: float, rocket_y: float):
        """Record current state."""
        self.cave_history.append(cave_slice)
        self.rocket_x_history.append(rocket_x)
        self.rocket_y_history.append(rocket_y)

        # Trim to max history
        if len(self.cave_history) > self.max_history:
            self.cave_history.pop(0)
            self.rocket_x_history.pop(0)
            self.rocket_y_history.pop(0)

    def render_vertical(self, height: int = 12, width: int = 20) -> str:
        """Render vertical history plot."""
        return render_vertical_history(
            self.cave_history,
            self.rocket_y_history,
            height=height,
            width=width
        )

    def render_horizontal(self, height: int = 8, width: int = 20) -> str:
        """Render horizontal history plot."""
        return render_horizontal_history(
            self.cave_history,
            self.rocket_x_history,
            height=height,
            width=width
        )

    def render_both(self, v_height: int = 12, h_height: int = 8, width: int = 20) -> str:
        """Render both plots stacked."""
        vertical = self.render_vertical(height=v_height, width=width)
        horizontal = self.render_horizontal(height=h_height, width=width)
        return vertical + "\n\n" + horizontal


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
    tracker = HistoryTracker(max_history=20)

    print("Initial state:")
    print(game.render())
    print()

    # Move in all 4 directions
    for _ in range(3):
        game.input_up()
        game.input_right()
        game.generate_random_cave()
        game.tick()
        # Record history
        rocket_slice_idx = 5
        if rocket_slice_idx < len(game.game.cave):
            tracker.record(
                game.game.cave[rocket_slice_idx],
                game.game.rocket.x,
                game.game.rocket.y
            )

    print("After UP+RIGHT (secretly: going long + buying):")
    print(game.render())
    print()

    # Now go down and left
    for _ in range(5):
        game.input_down()
        game.input_left()
        game.generate_random_cave()
        game.tick()
        # Record history
        rocket_slice_idx = 5
        if rocket_slice_idx < len(game.game.cave):
            tracker.record(
                game.game.cave[rocket_slice_idx],
                game.game.rocket.x,
                game.game.rocket.y
            )

    print("After DOWN+LEFT (secretly: going short + selling):")
    print(game.render())
    print()

    # Generate more history for better plots
    for _ in range(15):
        if game.game.state == GameState.PLAYING:
            # Alternate movements
            game.input_up() if _ % 2 == 0 else game.input_down()
            game.input_right() if _ % 3 == 0 else game.input_left()
            game.generate_random_cave()
            game.tick()
            # Record history
            rocket_slice_idx = 5
            if rocket_slice_idx < len(game.game.cave):
                tracker.record(
                    game.game.cave[rocket_slice_idx],
                    game.game.rocket.x,
                    game.game.rocket.y
                )

    print()
    print("=" * 40)
    print("HISTORY PLOTS")
    print("=" * 40)
    print()
    print(tracker.render_both())


if __name__ == "__main__":
    demo()

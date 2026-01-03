"""
Egg Trader - Pick up eggs, deposit them for profit!

A trading game disguised as an egg collection game.
Simple enough for an 8-year-old, teaches real trading intuition.

THE GAME:
    ┌─────────────────────────────────────────────────┐
    │░░░░░░░░░░░░░░ RESISTANCE ░░░░░░░░░░░░░░░░░░░░░░│
    │                                                 │
    │ 🟢 ← GREEN EGGS    →→→→→→→→    DEPOSIT → 🟢   │
    │    (go long here)              (sell here)      │
    │                    ◆ rocket                     │
    │ 🔴 ← RED EGGS      ←←←←←←←←    DEPOSIT → 🔴   │
    │    (go short here)             (cover here)     │
    │                                                 │
    │░░░░░░░░░░░░░░░ SUPPORT ░░░░░░░░░░░░░░░░░░░░░░░│
    └─────────────────────────────────────────────────┘

RULES:
- LEFT side: Pick up eggs (open positions)
  - Green eggs = Long positions (bet price goes UP)
  - Red eggs = Short positions (bet price goes DOWN)

- RIGHT side: Deposit eggs (close positions)
  - Green eggs deposited HIGHER than pickup = PROFIT 💰
  - Green eggs deposited LOWER than pickup = LOSS 💸
  - Red eggs deposited LOWER than pickup = PROFIT 💰
  - Red eggs deposited HIGHER than pickup = LOSS 💸

- TOP/BOTTOM: Support and Resistance
  - These shift as you fly across!
  - May trap you into a bad deposit

- The catch: There must be a BUYER (bid) where you want to deposit!
  - Order book shows where you CAN deposit
  - No bid at your price = can't deposit there

CONTROLS:
  ↑ UP    = fly up (toward higher prices)
  ↓ DOWN  = fly down (toward lower prices)
  ← LEFT  = fly left (toward pickup zone)
  → RIGHT = fly right (toward deposit zone)
  SPACE   = pick up / deposit egg

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EggType(Enum):
    """Type of egg (position)."""
    NONE = "none"
    GREEN = "green"  # Long position
    RED = "red"      # Short position


@dataclass
class Egg:
    """An egg (trading position)."""
    egg_type: EggType
    pickup_price: float      # Price level where picked up
    pickup_time: int         # Game tick when picked up
    quantity: float = 1.0    # Position size (e.g., 0.1 BTC, 5 XRP)


@dataclass
class DepositZone:
    """A place where eggs can be deposited (order book level)."""
    price_level: float       # Y position (0-1 normalized)
    depth: float             # How much can be deposited here (bid/ask size)
    is_bid: bool             # True = bid (for selling), False = ask (for buying)


@dataclass
class CaveState:
    """Current state of the cave."""
    support: float           # Bottom wall (0-1)
    resistance: float        # Top wall (0-1)
    bid_wall: float          # Left wall position
    ask_wall: float          # Right wall position
    bid_zones: List[DepositZone] = field(default_factory=list)  # Where you can sell
    ask_zones: List[DepositZone] = field(default_factory=list)  # Where you can buy


@dataclass
class Rocket:
    """Player's rocket."""
    x: float = 0.5           # 0=left (pickup), 1=right (deposit)
    y: float = 0.5           # 0=support, 1=resistance (price level)
    carrying: Optional[Egg] = None  # Current egg being carried


@dataclass
class Portfolio:
    """Track trading performance."""
    cash: float = 10000.0    # Starting cash
    total_pnl: float = 0.0   # Total profit/loss
    trades: List[dict] = field(default_factory=list)
    eggs_collected: int = 0
    eggs_deposited: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0


class EggTraderGame:
    """
    The Egg Trader game.

    Pick up eggs on the left, deposit on the right for profit!
    """

    def __init__(
        self,
        width: int = 40,
        height: int = 15,
        step_size: float = 0.05,
        egg_quantity: float = 1.0,  # Position size per egg
    ):
        self.width = width
        self.height = height
        self.step_size = step_size
        self.egg_quantity = egg_quantity

        self.rocket = Rocket()
        self.portfolio = Portfolio()
        self.cave = CaveState(
            support=0.2,
            resistance=0.8,
            bid_wall=0.1,
            ask_wall=0.9,
        )

        self.tick_count = 0
        self.game_over = False

        # Pickup zones on the left (where eggs spawn)
        self.green_egg_zone = (0.0, 0.15)   # x range for green eggs
        self.red_egg_zone = (0.0, 0.15)     # x range for red eggs

        # Deposit zones on the right
        self.deposit_zone = (0.85, 1.0)     # x range for depositing

        # Current price (for P&L calculation)
        self.current_price = 100.0
        self.price_history: List[float] = [100.0]

    def input_up(self):
        """Move up (toward higher prices)."""
        if not self.game_over:
            self.rocket.y = min(0.95, self.rocket.y + self.step_size)

    def input_down(self):
        """Move down (toward lower prices)."""
        if not self.game_over:
            self.rocket.y = max(0.05, self.rocket.y - self.step_size)

    def input_left(self):
        """Move left (toward pickup zone)."""
        if not self.game_over:
            self.rocket.x = max(0.0, self.rocket.x - self.step_size)

    def input_right(self):
        """Move right (toward deposit zone)."""
        if not self.game_over:
            self.rocket.x = min(1.0, self.rocket.x + self.step_size)

    def input_action(self):
        """Pick up or deposit an egg."""
        if self.game_over:
            return

        # In pickup zone (left side)?
        if self.rocket.x < 0.2 and self.rocket.carrying is None:
            self._pickup_egg()

        # In deposit zone (right side)?
        elif self.rocket.x > 0.8 and self.rocket.carrying is not None:
            self._deposit_egg()

    def _pickup_egg(self):
        """Pick up an egg based on vertical position."""
        # Upper half = green eggs (long), lower half = red eggs (short)
        if self.rocket.y > 0.5:
            egg_type = EggType.GREEN
        else:
            egg_type = EggType.RED

        self.rocket.carrying = Egg(
            egg_type=egg_type,
            pickup_price=self.rocket.y,  # Price level at pickup
            pickup_time=self.tick_count,
            quantity=self.egg_quantity,
        )
        self.portfolio.eggs_collected += 1

    def _deposit_egg(self):
        """Deposit the carried egg and calculate P&L."""
        egg = self.rocket.carrying
        if egg is None:
            return

        deposit_price = self.rocket.y
        pickup_price = egg.pickup_price

        # Calculate P&L based on egg type
        if egg.egg_type == EggType.GREEN:
            # Long: profit if deposit > pickup
            pnl = (deposit_price - pickup_price) * egg.quantity * self.current_price
        else:
            # Short: profit if deposit < pickup
            pnl = (pickup_price - deposit_price) * egg.quantity * self.current_price

        # Record trade
        self.portfolio.trades.append({
            'type': egg.egg_type.value,
            'pickup_price': pickup_price,
            'deposit_price': deposit_price,
            'quantity': egg.quantity,
            'pnl': pnl,
            'tick': self.tick_count,
        })

        self.portfolio.total_pnl += pnl
        self.portfolio.cash += pnl
        self.portfolio.eggs_deposited += 1

        if pnl > 0:
            self.portfolio.profitable_trades += 1
        else:
            self.portfolio.losing_trades += 1

        # Drop the egg
        self.rocket.carrying = None

    def tick(self) -> bool:
        """
        Advance game by one tick.
        Returns False if game over.
        """
        if self.game_over:
            return False

        self.tick_count += 1

        # Shift support/resistance (cave morphs)
        self._update_cave()

        # Check collision with support/resistance
        if self.rocket.y <= self.cave.support or self.rocket.y >= self.cave.resistance:
            self.game_over = True
            return False

        return True

    def _update_cave(self):
        """Update cave walls (support/resistance shift)."""
        import random

        # Random walk for support/resistance
        self.cave.support += random.uniform(-0.01, 0.015)
        self.cave.resistance += random.uniform(-0.015, 0.01)

        # Keep minimum gap
        min_gap = 0.4
        if self.cave.resistance - self.cave.support < min_gap:
            mid = (self.cave.resistance + self.cave.support) / 2
            self.cave.support = mid - min_gap / 2
            self.cave.resistance = mid + min_gap / 2

        # Clamp to valid range
        self.cave.support = max(0.05, min(0.4, self.cave.support))
        self.cave.resistance = max(0.6, min(0.95, self.cave.resistance))

        # Update price based on support/resistance midpoint
        mid_price = (self.cave.support + self.cave.resistance) / 2
        self.current_price = 80 + mid_price * 40  # Map to $80-$120 range
        self.price_history.append(self.current_price)

    def render(self) -> str:
        """Render the game as ASCII art."""
        lines = []

        # Header with P&L
        pnl_color = "+" if self.portfolio.total_pnl >= 0 else ""
        lines.append(f"═══ EGG TRADER ═══  P&L: ${pnl_color}{self.portfolio.total_pnl:.2f}")
        lines.append(f"Cash: ${self.portfolio.cash:.2f}  Eggs: {self.portfolio.eggs_deposited}/{self.portfolio.eggs_collected}")
        lines.append("")

        # Game area
        inner_w = self.width - 2
        inner_h = self.height - 2

        # Create grid
        grid = [[' ' for _ in range(inner_w)] for _ in range(inner_h)]

        # Draw support/resistance
        support_row = int((1 - self.cave.support) * (inner_h - 1))
        resistance_row = int((1 - self.cave.resistance) * (inner_h - 1))

        for col in range(inner_w):
            if resistance_row >= 0 and resistance_row < inner_h:
                grid[resistance_row][col] = '░'
            if support_row >= 0 and support_row < inner_h:
                grid[support_row][col] = '░'

        # Draw pickup zone (left side)
        pickup_col = 2
        for row in range(inner_h):
            y_norm = 1 - (row / (inner_h - 1))
            if y_norm > 0.5:
                grid[row][pickup_col] = '●'  # Green egg zone (shown as filled circle)
            else:
                grid[row][pickup_col] = '○'  # Red egg zone (shown as empty circle)

        # Draw deposit zone (right side)
        deposit_col = inner_w - 3
        for row in range(inner_h):
            grid[row][deposit_col] = '│'

        # Draw rocket
        rocket_col = int(self.rocket.x * (inner_w - 1))
        rocket_row = int((1 - self.rocket.y) * (inner_h - 1))
        rocket_col = max(0, min(inner_w - 1, rocket_col))
        rocket_row = max(0, min(inner_h - 1, rocket_row))

        if self.rocket.carrying:
            if self.rocket.carrying.egg_type == EggType.GREEN:
                rocket_char = '◆'  # Carrying green
            else:
                rocket_char = '◇'  # Carrying red
        else:
            rocket_char = '►'  # Empty rocket

        grid[rocket_row][rocket_col] = rocket_char

        # Build output
        lines.append("┌" + "─" * inner_w + "┐")
        lines.append("│" + " RESISTANCE (can't go higher)".ljust(inner_w)[:inner_w] + "│")

        for row in range(inner_h):
            # Add zone labels on sides
            y_norm = 1 - (row / (inner_h - 1))

            row_str = "".join(grid[row])
            lines.append("│" + row_str + "│")

        lines.append("│" + " SUPPORT (can't go lower)".ljust(inner_w)[:inner_w] + "│")
        lines.append("└" + "─" * inner_w + "┘")

        # Legend
        lines.append("")
        lines.append("PICKUP (left)          DEPOSIT (right)")
        lines.append("● = Long egg (up)      │ = Deposit zone")
        lines.append("○ = Short egg (down)")
        lines.append("")

        # Carrying indicator
        if self.rocket.carrying:
            egg = self.rocket.carrying
            egg_name = "LONG" if egg.egg_type == EggType.GREEN else "SHORT"
            lines.append(f"Carrying: {egg_name} egg from price {egg.pickup_price:.2f}")
            lines.append(f"Current price: {self.rocket.y:.2f}")
            if egg.egg_type == EggType.GREEN:
                pnl_preview = (self.rocket.y - egg.pickup_price) * egg.quantity * self.current_price
            else:
                pnl_preview = (egg.pickup_price - self.rocket.y) * egg.quantity * self.current_price
            lines.append(f"If deposited now: ${pnl_preview:+.2f}")
        else:
            lines.append("Not carrying any egg. Go LEFT to pick up!")

        lines.append("")
        lines.append("[Arrows] Move  [SPACE] Pickup/Deposit  [Q] Quit")

        if self.game_over:
            lines.append("")
            lines.append("*** CRASHED! Hit support/resistance ***")
            lines.append(f"Final P&L: ${self.portfolio.total_pnl:+.2f}")
            lines.append("[R] Restart")

        return "\n".join(lines)

    def reset(self):
        """Reset the game."""
        self.rocket = Rocket()
        self.cave = CaveState(
            support=0.2,
            resistance=0.8,
            bid_wall=0.1,
            ask_wall=0.9,
        )
        self.tick_count = 0
        self.game_over = False
        # Keep portfolio for cumulative tracking


def demo():
    """Demo the egg trader game."""
    print("=" * 50)
    print("EGG TRADER DEMO")
    print("=" * 50)
    print()

    game = EggTraderGame()

    print("Initial state:")
    print(game.render())
    print()

    # Simulate picking up a green egg
    print("Moving to pickup zone and getting a LONG egg...")
    for _ in range(10):
        game.input_left()
    for _ in range(5):
        game.input_up()
    game.input_action()  # Pickup

    print(game.render())
    print()

    # Move to deposit with profit
    print("Flying to deposit zone at higher price...")
    for _ in range(20):
        game.input_right()
        game.tick()
    for _ in range(3):
        game.input_up()  # Go higher for profit

    print(game.render())
    print()

    # Deposit
    print("Depositing egg...")
    game.input_action()

    print(game.render())


if __name__ == "__main__":
    demo()
